from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants, get_conversion_factor
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transport_analysis.utils import msd_fft_1d, msd_variance_1d, average_directions

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup


## Need to pass the entire the universe to get the com of the universe not just the atomg
class DiffusionCoefficientEinstein(AnalysisBase):
    """
        Class to calculate the diffusion_coefficient of a species.
        Note that the slope of the mean square displacement provides
        the diffusion coeffiecient. This implementation uses FFT to compute MSD faster
        as explained here: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
        Since we are using FFT, the sampling frequency (or) dump frequency of the simulation trajectory
        plays a critical role in the accuracy of the result.

        Particle positions are required to calculate the viscosity function. Thus
        you are limited to MDA trajectories that contain velocity information, e.g.
        GROMACS .trr files, H5MD files, etc. See the MDAnalysis documentation:
        https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#writers.

        Parameters
        ----------
        atomgroup : AtomGroup
            An MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`.
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        temp_avg : float (optional, default 300)
            Average temperature over the course of the simulation, in Kelvin.
        dim_type : {'xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'}
            Desired dimensions to be included in the viscosity calculation.
            Defaults to 'xyz'.
        linear_fit_window : tuple of int (optional)
            A tuple of two integers specifying the start and end lag-time for
            the linear fit of the viscosity function. If not provided, the
            linear fit is not performed and viscosity is not calculated.

        Attributes
        ----------
        atomgroup : :class:`~MDAnalysis.core.groups.AtomGroup`
            The atoms to which this analysis is applied
        dim_fac : int
            Dimensionality :math:`d` of the viscosity computation
        start : Optional[int]
            The first frame of the trajectory used to compute the analysis.
        stop : Optional[int]
            The frame to stop at for the analysis.
        step : Optional[int]
            Number of frames to skip between each analyzed frame.
        n_frames : int
            Number of frames analysed in the trajectory.
        n_particles : int
            Number of particles viscosity was calculated over.
        """

    def __init__(
        self,
        all_atoms: "AtomGroup",
        specie: "AtomGroup",
        temp_avg: float = 298.0,
        dim_type: str = "xyz",
        linear_fit_window: tuple[int, int] = None,
        **kwargs,
    ) -> None:
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(all_atoms.universe.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results
        if isinstance(all_atoms, UpdatingAtomGroup):
            raise TypeError(
                "UpdatingAtomGroups are not valid for diffusion computation"
            )
        self.temp_avg = temp_avg
        self.linear_fit_window = linear_fit_window

        self.all_atoms = all_atoms
        self.specie = specie
        self.specie_num_atoms = len(self.specie)
        self.num_atoms_per_species = int(len(specie) / len(specie.residues))
        self.mass_per_species = sum(specie[:self.num_atoms_per_species].masses)

        self.dim_type = dim_type.lower()
        self._dim, self.dim_fac = self._parse_dim_type(self.dim_type)

        if self.specie_num_atoms % self.num_atoms_per_species != 0:
            raise TypeError("Some species are fragmented. Invalid AtomGroup")

    def _prepare(self):
        """
        Set up mass and position arrays before the analysis loop begins
        """

        self._volumes = np.zeros((self.n_frames))
        self._msds = np.zeros((self.n_frames, self._dim))
        self._msds_var = np.zeros((self.n_frames, self._dim))
        self._lii_self = np.zeros((self.n_frames))
        self.weights = np.ones((self.n_frames))
        self._coms = np.zeros((self.n_frames, self._dim))
        self._times = np.zeros((self.n_frames))

        self._masses = self.all_atoms.masses
        self.specie_masses = self.specie.masses

        # 3D arrays of frames x particles x dimensions
        # positions
        self._positions = np.zeros((self.n_frames, int(self.specie_num_atoms), self._dim))
        self._specie_positions = np.zeros(
            (
                self.n_frames,
                int(self.specie_num_atoms / self.num_atoms_per_species),
                self._dim,
            )
        )
        self.boltzmann =  constants["Boltzmann_constant"] * get_conversion_factor('energy', 'kJ/mol', 'J')
        self.ps_to_s = get_conversion_factor('time','ps', 's')
        self.A_to_m = get_conversion_factor('speed','A/ps', 'm/s')*self.ps_to_s
        self.conversion_factor = (1 / (self.ps_to_s * self.A_to_m))

    def _single_frame(self):
        """
        Constructs arrays of velocities and positions
        for viscosity computation.
        """
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # trajectory must have velocity and position information
        if not (self._ts.has_positions and self._ts.volume != 0):
            raise NoDataError(
                "Einstein diffusion computation requires positions, and box volume in the trajectory"
            )

        # fill volume array
        self._volumes[self._frame_index] = self._ts.volume
        self._times[self._frame_index] = self._ts.time
        # set shape of position array

        self._positions[self._frame_index] = self.specie.positions
        self._coms[self._frame_index] = self.all_atoms.center_of_mass(wrap=False)
        self._specie_positions[self._frame_index] = (
            (
                np.multiply(
                    np.repeat(self.specie_masses, self._dim, axis=0).reshape(-1, self._dim),
                    self._positions[self._frame_index],
                )
                / self.mass_per_species
            )
            .reshape(-1, int(self.num_atoms_per_species), self._dim)
            .sum(axis=1)
        )
        self._specie_positions[self._frame_index] = (
            self._specie_positions[self._frame_index]
            - self._coms[self._frame_index]
        )

    def _conclude(self):
        """
        Calculates the diffusion coefficient via the fft.
        """
        for specie_id in range(len(self.specie.residues)):
            for i in range(self._dim):
                msd_temp = msd_fft_1d(
                    np.array(self._specie_positions[:, specie_id, :][:, i])
                )
                self._msds[:, i] += msd_temp
                self._msds_var[:, i] += msd_variance_1d(
                    np.array(self._specie_positions[:, specie_id, :][:, i]),
                    self._msds[:, i],
                )

        self._lii_self = average_directions(self._msds, self._dim)*self.conversion_factor / (
            2 * self.boltzmann * self.temp_avg * self._volumes
        )
        self.weights = np.sqrt(
            np.abs(1 / average_directions(self._msds_var, self._dim))
        )
        self.weights /= np.sum(self.weights)

        self.lii_self = None
        beta = None
        if self.linear_fit_window:
            lii_self, _ , _, _, _ = stats.linregress(
                self._times[self.linear_fit_window[0] : self.linear_fit_window[1]],
                self._lii_self[self.linear_fit_window[0] : self.linear_fit_window[1]],
            )
            fit_slope = np.gradient(
                np.log(
                    self._lii_self[
                        self.linear_fit_window[0] : self.linear_fit_window[1]
                    ]
                )[1:],
                np.log(
                    self._times[self.linear_fit_window[0] : self.linear_fit_window[1]]
                    - self._times[0]
                )[1:],
            )
            beta = np.nanmean(np.array(fit_slope))

        else:
            _ , lii_self = np.polynomial.polynomial.polyfit(
                self._times[1:], self._lii_self[1:], 1, w=self.weights[1:]
            )


        self.conc = float(len(self.specie.residues)) / (
            self._volumes.mean() * (self.A_to_m**3)
        )

        if self.linear_fit_window:
            self.results.linearity = beta

        self.results.lii_self = lii_self
        self.results.diffusion_coefficient = (lii_self) * (self.boltzmann * self.temp_avg / self.conc)

    def plot_linear_fit(self, plot_type="log-log"):
        """
        Plot the mead square displacement vs time and the fit region in the log-log scale
        """
        plt.plot(
            self._times,
            self._lii_self,
            label="- -",
            color="tab:green"
        )
        i = int(len(self._lii_self) / 5)
        slope_guess = (self._lii_self[i] - self._lii_self[5]) / (self._times[i] - self._times[5])
        plt.plot(self._times, self._times * slope_guess * 2, "k--", alpha=0.5, label = "Slope = 1")  
        plt.set_ylabel("MSD -- (S*s/m)")
        plt.set_xlabel("Time (ps)")
        plt.set_title("MSD Self")
        plt.grid()
        plt.legend()
        if self.linear_fit_window:
            plt.axvline(x=self._times[self.linear_fit_window[0]], color='r', linestyle='--', linewidth=2)
            plt.axvline(x=self._times[self.linear_fit_window[1]], color='r', linestyle='--', linewidth=2)
        if plot_type == "log-log":
            plt.set_xscale("log")
            plt.set_yscale("log")
        elif plot_type == "linear-linear":
            plt.set_xscale("linear")
            plt.set_yscale("linear")
            
        plt.tight_layout()
        plt.show()
