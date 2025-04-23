from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants, get_conversion_factor
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import stats
from transport_analysis.utils import cross_corr, average_directions
from transport_analysis.utils import msd_fft_cross_1d, msd_variance_cross_1d
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup


## Need to pass the entire the universe to get the com of the universe not just the atomg
class ConductivityKubo(AnalysisBase):
    """
        Class to calculate the conductivity from MD simulaiton. This implementation uses FFT to compute autocorrelation faster
        as explained here: https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        Since we are using FFT, the sampling frequency (or) dump frequency of the simulation trajectory
        plays a critical role in the accuracy of the result.

        Particle velocities are required to calculate the velocity-velocity autocorrelation function. Thus
        you are limited to MDA trajectories that contain velocity information, e.g.
        GROMACS .trr files, H5MD files, etc. See the MDAnalysis documentation:
        https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#writers.

        Parameters
        ----------
        all_atoms : AtomGroup
            An MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`.
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        cations : :class:`~MDAnalysis.core.groups.AtomGroup`
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        anions : :class:`~MDAnalysis.core.groups.AtomGroup`
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        temp_avg : float (optional, default 300)
            Average temperature over the course of the simulation, in Kelvin.
        dim_type : {'xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'}
            Desired number of dimensions to be included in the velocity-velocity correlation calculation.
            Defaults to 'xyz'.

        Attributes
        ----------
        atomgroup : :class:`~MDAnalysis.core.groups.AtomGroup`
            The atoms to which this analysis is applied
        cations : :class:`~MDAnalysis.core.groups.AtomGroup`
            The cations to which this analysis is applied
        anions : :class:`~MDAnalysis.core.groups.AtomGroup`
            The anions to which this analysis is applied
        dim_fac : int
            Dimensionality :math:`d` of the conductivity computation.
        results.conductivity : float
            The conductivity of the cation-anion in the solvent in S/m.
        results.cation_transference : float
            The cation transference number in the centre of mass frame.
        results.anion_transference : float
            The anion transference number in the centre of mass frame.
        start : Optional[int]
            The first frame of the trajectory used to compute the analysis.
        stop : Optional[int]
            The frame to stop at for the analysis.
        step : Optional[int]
            Number of frames to skip between each analyzed frame.
        n_frames : int
            Number of frames analysed in the trajectory.
        n_particles : int
            Total number of particles in the trajectory.
    """

    def __init__(
        self,
        all_atoms: "AtomGroup",
        cations: "AtomGroup",
        anions: "AtomGroup",
        temp_avg: float = 298.0,
        dim_type: str = "xyz",
        cation_charge: int=1,
        anion_charge: int=-1,
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
                "UpdatingAtomGroups are not valid for conductivity computation"
            )
        if isinstance(cations, UpdatingAtomGroup):
            raise TypeError(
                "UpdatingAtomGroups are not valid for conductivity computation"
            )
        if isinstance(anions, UpdatingAtomGroup):
            raise TypeError(
                "UpdatingAtomGroups are not valid for conductivity computation"
            )
        self.temp_avg = temp_avg

        self.all_atoms = all_atoms
        self.cations = cations
        self.anions = anions
        self.cation_num_atoms_per_species = int(len(cations) / len(cations.residues))
        self.anion_num_atoms_per_species = int(len(anions) / len(anions.residues))
        self.cation_mass_per_species = sum(cations[:self.cation_num_atoms_per_species].masses)
        self.anion_mass_per_species = sum(anions[:self.anion_num_atoms_per_species].masses)
        self.cation_charge = cation_charge
        self.anion_charge = anion_charge
        
        self.dim_type = dim_type.lower()
        self._dim, self.dim_fac = self._parse_dim_type(self.dim_type)
        self.cation_num_atoms = len(self.cations)
        self.anion_num_atoms = len(self.anions)

        if (self.cation_num_atoms % self.cation_num_atoms_per_species != 0) or (
            self.anion_num_atoms % self.anion_num_atoms_per_species != 0
        ):
            raise TypeError("Some species are fragmented. Invalid AtomGroup")


    @staticmethod
    def _parse_dim_type(dim_str):
        r"""Sets up the desired dimensionality of the calculation."""
        keys = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }

        try:
            _dim = keys[dim_str]
        except KeyError:
            raise ValueError(
                "invalid dim_type: {} specified, please specify one of xyz, "
                "xy, xz, yz, x, y, z".format(dim_str)
            )

        return _dim, len(_dim)
    
    @staticmethod
    def _get_cutoff(conductivity, times):
        A_guess = 0
        tau_guess = 1
        C_guess = 0
        # Fit the data to an exponential function
        def _exp_fit(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        p0 = [A_guess, tau_guess, C_guess]

        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])  # A, tau, C all positive

        params, _ = curve_fit(_exp_fit, times[:-1], conductivity, p0=p0, bounds=bounds)

        # Extract fitted parameters
        A_fit, tau_fit, C_fit = params
        # Calculate the cutoff step with double precision
        # The cutoff time is the time at which the fitted function is equal to 0
        cutoff_time = -np.log(10**(-16))*tau_fit
        return np.argmin(np.abs(times - cutoff_time))

    def _prepare(self):
        """
        Set up volume, mass, velocity, velocity-velocity auto-correlation arrays
        before the analysis 
        """
        # 2D viscosity array of frames x particles

        self._volumes = np.zeros((self.n_frames))

        self._acf_cation_cation = np.zeros((self.n_frames, self._dim))
        self._acf_cation_anion = np.zeros((self.n_frames, self._dim))
        self._acf_anion_anion = np.zeros((self.n_frames, self._dim))

        self._l_cation_cation = np.zeros((self.n_frames - 1, self._dim))
        self._l_cation_anion = np.zeros((self.n_frames - 1, self._dim))
        self._l_anion_anion = np.zeros((self.n_frames - 1, self._dim))
        self._lij_cation_cation = np.zeros((self.n_frames - 1))
        self._lij_cation_anion = np.zeros((self.n_frames - 1))
        self._lij_anion_anion = np.zeros((self.n_frames - 1))

        self._cond = np.zeros((self.n_frames - 1))

        self._coms_velocity = np.zeros((self.n_frames, self._dim))
        self._times = np.zeros((self.n_frames))

        self._cation_masses = self.cations.masses
        self._anion_masses = self.anions.masses
        self._all_atom_masses = self.all_atoms.masses

        # 3D arrays of frames x particles x dimensions
        # positions
        self._cation_atom_velocities = np.zeros(
            (self.n_frames, int(self.cation_num_atoms), self._dim)
        )
        self._anion_atom_velocities = np.zeros(
            (self.n_frames, int(self.anion_num_atoms), self._dim)
        )

        self._cation_specie_velocities = np.zeros(
            (
                self.n_frames,
                int(self.cation_num_atoms / self.cation_num_atoms_per_species),
                self._dim,
            )
        )

        self._anion_specie_velocities = np.zeros(
            (
                self.n_frames,
                int(self.anion_num_atoms / self.anion_num_atoms_per_species),
                self._dim,
            )
        )
        self.boltzmann =  constants["Boltzmann_constant"] * get_conversion_factor('energy', 'kJ/mol', 'J')
        self.ps_to_s = get_conversion_factor('time','ps', 's')
        self.e_to_C = get_conversion_factor('charge','e', 'C')
        self.A_to_m = get_conversion_factor('speed','A/ps', 'm/s')*self.ps_to_s
        self.conversion_factor = (self.e_to_C * self.e_to_C)* (1 / (self.ps_to_s * self.A_to_m))

    def _single_frame(self):
        """
        Constructs arrays of atom velocities and specie velocities
        """
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # trajectory must have velocity and position information
        if not (self._ts.has_velocities and self._ts.volume != 0):
            raise NoDataError(
                "Green Kubo conductivity computation requires velocities, and box volume in the trajectory"
            )

        # fill volume array
        self._volumes[self._frame_index] = self._ts.volume
        self._times[self._frame_index] = self._ts.time
        # set shape of position array

        self._cation_atom_velocities[self._frame_index] = self.cations.velocities
        self._anion_atom_velocities[self._frame_index] = self.anions.velocities
        self._coms_velocity[self._frame_index] = (
            np.multiply(
                np.repeat(self._all_masses, 3, axis=0).reshape(-1, 3),
                self.all_atoms.velocities,
            ).sum(axis=0)
            / self._all_atom_masses.sum()
        )

        self._cation_specie_velocities[self._frame_index] = (
            (
                np.multiply(
                    np.repeat(self._cation_masses, self._dim, axis=0).reshape(
                        -1, self._dim
                    ),
                    self._cation_atom_velocities[self._frame_index],
                )
                / self.cation_mass_per_species
            )
            .reshape(-1, int(self.cation_num_atoms_per_species), self._dim)
            .sum(axis=1)
        )
        self._anion_specie_velocities[self._frame_index] = (
            (
                np.multiply(
                    np.repeat(self._anion_masses, self._dim, axis=0).reshape(
                        -1, self._dim
                    ),
                    self._anion_atom_velocities[self._frame_index],
                )
                / self.anion_mass_per_species
            )
            .reshape(-1, int(self.anion_num_atoms_per_species), self._dim)
            .sum(axis=1)
        )

        self._cation_specie_velocities[self._frame_index] = (
            self._cation_specie_velocities[self._frame_index]
            - self._coms_velocity[self._frame_index]
        )
        self._anion_specie_velocities[self._frame_index] = (
            self._anion_specie_velocities[self._frame_index]
            - self._coms_velocity[self._frame_index]
        )

    def _conclude(self):
        """
        Calculates the conductivity coefficient via the fft.
        """
        cation_summed_velocities = np.sum(self._cation_specie_velocities, axis=1)
        anion_summed_velocities = np.sum(self._anion_specie_velocities, axis=1)
        for i in range(self._dim):
            self._acf_cation_cation[:, i] = cross_corr(
                cation_summed_velocities[:, i], cation_summed_velocities[:, i]
            )
            self._acf_anion_anion[:, i] = cross_corr(
                anion_summed_velocities[:, i], anion_summed_velocities[:, i]
            )
            self._acf_cation_anion[:, i] = cross_corr(
                cation_summed_velocities[:, i], anion_summed_velocities[:, i]
            )
        for i in range(self._dim):
            self._l_cation_cation[:, i] = integrate.cumulative_trapezoid(
                self._acf_cation_cation[:, i], self._times[:, i]
            )
            self._l_anion_anion[:, i] = integrate.cumulative_trapezoid(
                self._acf_anion_anion[:, i], self._times[:, i]
            )
            self._l_cation_anion[:, i] = integrate.cumulative_trapezoid(
                self._acf_cation_anion[:, i], self.times[:, i]
            )

        self._lij_cation_cation = average_directions(self._l_cation_cation, self._dim)*self.conversion_factor / (
            2 * self.boltzmann * self.temp_avg * self._volumes
        )
        self._lij_cation_anion = average_directions(self._l_anion_anion, self._dim)*self.conversion_factor / (
            2 * self.boltzmann * self.temp_avg * self._volumes
        )
        self._lij_anion_anion = average_directions(self._l_cation_anion, self._dim)*self.conversion_factor / (
            2 * self.boltzmann * self.temp_avg * self._volumes
        )


        self._cond = (
            (self.cation_charge**2) * self._lij_cation_cation
            + (self.anion_charge**2) * self._lij_anion_anion
            + (2 * self.cation_charge * self.anion_charge) * self._lij_cation_anion
        )

        if self.cutoff_step is None:
            self.cutoff_step = self._get_cutoff(self._cond, self._times)


        self.results.conductivity = self._cond[self.cutoff_step]

        self.results.cation_transference = (
            (self.cation_charge**2) * self._lij_cation_cation[self.cutoff_step]
            + (self.cation_charge * self.anion_charge)
            * self._lij_cation_anion[self.cutoff_step]
        ) / self._cond[self.cutoff_step]

        self.results.anion_transference = (
            (self.anion_charge**2) * self._lij_anion_anion[self.cutoff_step]
            + (self.cation_charge * self.anion_charge)
            * self._lij_cation_anion[self.cutoff_step]
        ) / self._cond[self.cutoff_step]

    def plot_acf(self):
        """
        Plot the velocity-velocity autocorrelation function vs lag-time as three subplots
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(
            self._times[: self.cutoff_step],
            self._acf_cation_cation[: self.cutoff_step],
            label="+ +",
            color="tab:blue"
        )
        axs[0].set_ylabel("ACF ++")
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(
            self._times[: self.cutoff_step],
            self._acf_cation_anion[: self.cutoff_step],
            label="+ -",
            color="tab:orange"
        )
        axs[1].set_ylabel("ACF +-")
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(
            self._times[: self.cutoff_step],
            self._acf_anion_anion[: self.cutoff_step],
            label="- -",
            color="tab:green"
        )
        axs[2].set_ylabel("ACF ₋₋")
        axs[2].set_xlabel("Time (ps)")
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.show()

    
    def plot_onsager_coeff(self):
        """
        Plot the Onsager coefficients vs lag-time as three subplots
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(
            self._times[: self.cutoff_step],
            self._lij_cation_cation[: self.cutoff_step],
            label="+ +",
            color="tab:blue"
        )
        axs[0].set_ylabel("L++ (S/m)")
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(
            self._times[: self.cutoff_step],
            self._lij_cation_anion[: self.cutoff_step],
            label="+ -",
            color="tab:orange"
        )
        axs[1].set_ylabel("L+- (S/m)")
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(
            self._times[: self.cutoff_step],
            self._lij_anion_anion[: self.cutoff_step],
            label="- -",
            color="tab:green"
        )
        axs[2].set_ylabel("L-- (S/m)")
        axs[2].set_xlabel("Time (ps)")
        axs[2].grid()
        axs[2].legend()

        plt.tight_layout()
        plt.show()


class ConductivityEinstein(AnalysisBase):
    """
        Class to calculate the conductivity from MD simulation. This implementation uses FFT to compute MSD faster
        as explained here: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
        Since we are using FFT, the sampling frequency (or) dump frequency of the simulation trajectory
        plays a critical role in the accuracy of the result.

        Particle positions are required to calculate the mean square displacement function. Thus
        you are limited to MDA trajectories that contain position information, e.g.
        GROMACS .trr files, H5MD files, etc. See the MDAnalysis documentation:
        https://docs.mdanalysis.org/stable/documentation_pages/coordinates/init.html#writers.

        Parameters
        ----------
        all_atoms : AtomGroup
            An MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`.
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        cations : :class:`~MDAnalysis.core.groups.AtomGroup`
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        anions : :class:`~MDAnalysis.core.groups.AtomGroup`
            Note that :class:`UpdatingAtomGroup` instances are not accepted.
        temp_avg : float (optional, default 300)
            Average temperature over the course of the simulation, in Kelvin.
        dim_type : {'xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'}
            Desired number of dimensions to be included in the velocity-velocity correlation calculation.
            Defaults to 'xyz'.

        Attributes
        ----------
        atomgroup : :class:`~MDAnalysis.core.groups.AtomGroup`
            The atoms to which this analysis is applied
        cations : :class:`~MDAnalysis.core.groups.AtomGroup`
            The cations to which this analysis is applied
        anions : :class:`~MDAnalysis.core.groups.AtomGroup`
            The anions to which this analysis is applied
        dim_fac : int
            Dimensionality :math:`d` of the conductivity computation.
        results.conductivity : float
            The conductivity of the cation-anion in the solvent in S/m.
        results.cation_transference : float
            The cation transference number in the centre of mass frame.
        results.anion_transference : float
            The anion transference number in the centre of mass frame.
        start : Optional[int]
            The first frame of the trajectory used to compute the analysis.
        stop : Optional[int]
            The frame to stop at for the analysis.
        step : Optional[int]
            Number of frames to skip between each analyzed frame.
        n_frames : int
            Number of frames analysed in the trajectory.
        n_particles : int
            Total number of particles in the trajectory.
    """

    def __init__(
        self,
        all_atoms: "AtomGroup",
        cations: "AtomGroup",
        anions: "AtomGroup",
        temp_avg: float = 298.0,
        dim_type: str = "xyz",
        linear_fit_window: tuple[int, int] = None,
        cation_charge: int=1,
        anion_charge: int=-1,
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
        self.cations = cations
        self.anions = anions
        self.cation_num_atoms_per_species = int(len(cations) / len(cations.residues))
        self.anion_num_atoms_per_species = int(len(anions) / len(anions.residues))
        self.cation_mass_per_species = sum(cations[:self.cation_num_atoms_per_species].masses)
        self.anion_mass_per_species = sum(anions[:self.anion_num_atoms_per_species].masses)
        self.cation_charge = cation_charge
        self.anion_charge = anion_charge

        self.dim_type = dim_type.lower()
        self._dim, self.dim_fac = self._parse_dim_type(self.dim_type)
        self.cation_num_atoms = len(self.cations)
        self.anion_num_atoms = len(self.anions)

        if (self.cation_num_atoms % self.cation_num_atoms_per_species != 0) or (
            self.anion_num_atoms % self.anion_num_atoms_per_species != 0
        ):
            raise TypeError("Some species are fragmented. Invalid AtomGroup")

    @staticmethod
    def _parse_dim_type(dim_str):
        r"""Sets up the desired dimensionality of the calculation."""
        keys = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }

        try:
            _dim = keys[dim_str]
        except KeyError:
            raise ValueError(
                "invalid dim_type: {} specified, please specify one of xyz, "
                "xy, xz, yz, x, y, z".format(dim_str)
            )

        return _dim, len(_dim)

    def _prepare(self):
        """
        Set up viscosity, mass, velocity, and position arrays
        before the analysis loop begins
        """
        # 2D viscosity array of frames x particles

        self._volumes = np.zeros((self.n_frames))

        self._msds_cation_cation = np.zeros((self.n_frames, self._dim))
        self._msds_cation_anion = np.zeros((self.n_frames, self._dim))
        self._msds_anion_anion = np.zeros((self.n_frames, self._dim))

        self._msds_cation_cation_var = np.zeros((self.n_frames, self._dim))
        self._msds_cation_anion_var = np.zeros((self.n_frames, self._dim))
        self._msds_anion_anion_var = np.zeros((self.n_frames, self._dim))
        self._cond = np.zeros((self.n_frames))
        self._cond_var = np.zeros((self.n_frames))

        self._lij_cation_cation = np.zeros((self.n_frames))
        self._lij_cation_anion = np.zeros((self.n_frames))
        self._lij_anion_anion = np.zeros((self.n_frames))

        self._lij_cation_cation_weights = np.ones((self.n_frames))
        self._lij_cation_anion_weights = np.ones((self.n_frames))
        self._lij_anion_anion_weights = np.ones((self.n_frames))

        self._coms = np.zeros((self.n_frames, self._dim))
        self._times = np.zeros((self.n_frames))

        self._cation_masses = self.cations.masses
        self._anion_masses = self.anions.masses

        # 3D arrays of frames x particles x dimensions
        # positions
        self._cation_atom_positions = np.zeros(
            (self.n_frames, int(self.cation_num_atoms), self._dim)
        )
        self._anion_atom_positions = np.zeros(
            (self.n_frames, int(self.anion_num_atoms), self._dim)
        )

        self._cation_specie_positions = np.zeros(
            (
                self.n_frames,
                int(self.cation_num_atoms / self.cation_num_atoms_per_species),
                self._dim,
            )
        )

        self._anion_specie_positions = np.zeros(
            (
                self.n_frames,
                int(self.anion_num_atoms / self.anion_num_atoms_per_species),
                self._dim,
            )
        )
        self.boltzmann =  constants["Boltzmann_constant"] * get_conversion_factor('energy', 'kJ/mol', 'J')
        self.ps_to_s = get_conversion_factor('time','ps', 's')
        self.e_to_C = get_conversion_factor('charge','e', 'C')
        self.A_to_m = get_conversion_factor('speed','A/ps', 'm/s')*self.ps_to_s
        self.conversion_factor = (self.e_to_C * self.e_to_C)* (1 / (self.ps_to_s * self.A_to_m))

    def _single_frame(self):
        """
        Constructs arrays of positions
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

        self._cation_atom_positions[self._frame_index] = self.cations.positions
        self._anion_atom_positions[self._frame_index] = self.anions.positions
        self._coms[self._frame_index] = self.all_atoms.center_of_mass(wrap=False)

        self._cation_specie_positions[self._frame_index] = (
            (
                np.multiply(
                    np.repeat(self._cation_masses, self._dim, axis=0).reshape(
                        -1, self._dim
                    ),
                    self._cation_atom_positions[self._frame_index],
                )
                / self.cation_mass_per_species
            )
            .reshape(-1, int(self.cation_num_atoms_per_species), self._dim)
            .sum(axis=1)
        )
        self._anion_specie_positions[self._frame_index] = (
            (
                np.multiply(
                    np.repeat(self._anion_masses, self._dim, axis=0).reshape(
                        -1, self._dim
                    ),
                    self._anion_atom_positions[self._frame_index],
                )
                / self.anion_mass_per_species
            )
            .reshape(-1, int(self.anion_num_atoms_per_species), self._dim)
            .sum(axis=1)
        )

        self._cation_specie_positions[self._frame_index] = (
            self._cation_specie_positions[self._frame_index]
            - self._coms[self._frame_index]
        )
        self._anion_specie_positions[self._frame_index] = (
            self._anion_specie_positions[self._frame_index]
            - self._coms[self._frame_index]
        )

    def _conclude(self):
        """
        Calculates the conductivity coefficient via the fft.
        """
        cation_summed_positions = np.sum(self._cation_specie_positions, axis=1)
        anion_summed_positions = np.sum(self._anion_specie_positions, axis=1)

        self._msds_cation_cation = np.transpose(
            [
                msd_fft_cross_1d(
                    cation_summed_positions[:, i], cation_summed_positions[:, i]
                )
                for i in range(self._dim)
            ]
        )
        self._msds_cation_cation_var = np.transpose(
            [
                msd_variance_cross_1d(
                    cation_summed_positions[:, i],
                    cation_summed_positions[:, i],
                    self._msds_cation_cation[:, i],
                )
                for i in range(self._dim)
            ]
        )
        self._msds_anion_anion = np.transpose(
            [
                msd_fft_cross_1d(
                    anion_summed_positions[:, i], anion_summed_positions[:, i]
                )
                for i in range(self._dim)
            ]
        )
        self._msds_anion_anion_var = np.transpose(
            [
                msd_variance_cross_1d(
                    anion_summed_positions[:, i],
                    anion_summed_positions[:, i],
                    self._msds_anion_anion[:, i],
                )
                for i in range(self._dim)
            ]
        )
        self._msds_cation_anion = np.transpose(
            [
                msd_fft_cross_1d(
                    cation_summed_positions[:, i], anion_summed_positions[:, i]
                )
                for i in range(self._dim)
            ]
        )
        self._msds_cation_anion_var = np.transpose(
            [
                msd_variance_cross_1d(
                    cation_summed_positions[:, i],
                    anion_summed_positions[:, i],
                    self._msds_cation_anion[:, i],
                )
                for i in range(self._dim)
            ]
        )
        self._lij_cation_cation = average_directions(
            self._msds_cation_cation, self._dim
        )*self.conversion_factor / (2 * self.boltzmann * self.temp_avg * self._volumes)
        self._lij_cation_anion = average_directions(
            self._msds_cation_anion, self._dim
        )*self.conversion_factor / (2 * self.boltzmann * self.temp_avg * self._volumes)
        self._lij_anion_anion = average_directions(
            self._msds_anion_anion, self._dim
        )*self.conversion_factor / (2 * self.boltzmann * self.temp_avg * self._volumes)


        self._lij_cation_cation_weights = np.sqrt(
            np.abs(1 / average_directions(self._msds_cation_cation_var, self._dim))
        )
        self._lij_cation_cation_weights /= np.sum(self._lij_cation_cation_weights)

        self._lij_cation_anion_weights = np.sqrt(
            np.abs(1 / average_directions(self._msds_cation_anion_var, self._dim))
        )
        self._lij_cation_anion_weights /= np.sum(self._lij_cation_anion_weights)

        self._lij_anion_anion_weights = np.sqrt(
            np.abs(1 / average_directions(self._msds_anion_anion_var, self._dim))
        )
        self._lij_anion_anion_weights /= np.sum(self._lij_anion_anion_weights)

        self.cond = None
        self.lij_cation_cation = None
        self.lij_cation_anion = None
        self.lij_anion_anion = None
        beta = None
        if self.linear_fit_window:
            self.lij_cation_cation, _, _, _, _ = stats.linregress(
                self._times[self.linear_fit_window[0] : self.linear_fit_window[1]],
                self._lij_cation_cation[
                    self.linear_fit_window[0] : self.linear_fit_window[1]
                ],
            )
            self.lij_cation_anion, _, _, _, _ = stats.linregress(
                self._times[self.linear_fit_window[0] : self.linear_fit_window[1]],
                self._lij_cation_anion[
                    self.linear_fit_window[0] : self.linear_fit_window[1]
                ],
            )
            self.lij_anion_anion, _, _, _, _ = stats.linregress(
                self._times[self.linear_fit_window[0] : self.linear_fit_window[1]],
                self._lij_anion_anion[
                    self.linear_fit_window[0] : self.linear_fit_window[1]
                ],
            )
            self.cond = (
                (self.cation_charge**2) * self.lij_cation_cation
                + (self.anion_charge**2) * self.lij_anion_anion
                + (2 * self.cation_charge * self.anion_charge) * self.lij_cation_anion
            )

            self._cond = (
                (self.cation_charge**2) * self._lij_cation_cation
                + (self.anion_charge**2) * self._lij_anion_anion
                + (2 * self.cation_charge * self.anion_charge) * self._lij_cation_anion
            )
            fit_slope = np.gradient(
                np.log(
                    self._cond[self.linear_fit_window[0] : self.linear_fit_window[1]]
                )[1:],
                np.log(
                    self._times[self.linear_fit_window[0] : self.linear_fit_window[1]]
                    - self._times[0]
                )[1:],
            )
            beta = np.nanmean(np.array(fit_slope))

        else:
            _, self.lij_cation_cation = np.polynomial.polynomial.polyfit(
                self._times[1:],
                self._lij_cation_cation[1:],
                1,
                w=self._lij_cation_cation_weights[1:],
            )
            _, self.lij_cation_anion = np.polynomial.polynomial.polyfit(
                self._times[1:],
                self._lij_cation_anion[1:],
                1,
                w=self._lij_cation_anion_weights[1:],
            )
            _, self.lij_anion_anion = np.polynomial.polynomial.polyfit(
                self._times[1:],
                self._lij_anion_anion[1:],
                1,
                w=self._lij_anion_anion_weights[1:],
            )
            self.cond = (
                (self.cation_charge**2) * self.lij_cation_cation
                + (self.anion_charge**2) * self.lij_anion_anion
                + (2 * self.cation_charge * self.anion_charge) * self.lij_cation_anion
            )
        if self.linear_fit_window:
            self.results.linearity = beta

        self.results.conductivity = self.cond
        self.results.cation_transference = (
            (self.cation_charge**2) * self.lij_cation_cation
            + (self.cation_charge * self.anion_charge) * self.lij_cation_anion
        ) / self.cond
        self.results.anion_transference= (
            (self.anion_charge**2) * self.lij_anion_anion
            + (self.cation_charge * self.anion_charge) * self.lij_cation_anion
        ) / self.cond

    def plot_linear_fit(self, plot_type="log-log"):
        """
        Plot the mead square displacement vs time and the fit region in the log-log scale
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axs[0].plot(
            self._times,
            self._lij_cation_cation,
            label="+ +",
            color="tab:blue"
        )
        i = int(len(self._lij_cation_cation) / 5)
        slope_guess = (self._lij_cation_cation[i] - self._lij_cation_cation[5]) / (self._times[i] - self._times[5])
        axs[0].plot(self._times, self._times * slope_guess * 2, "k--", alpha=0.5, label = "Slope = 1")  
        axs[0].set_ylabel("MSD ++ (S*s/m)")
        axs[0].set_xlabel("Time (ps)")
        axs[0].set_title("Cation-Cation")
        axs[0].grid()
        axs[0].legend()
        if self.linear_fit_window:
            axs[0].axvline(x=self._times[self.linear_fit_window[0]], color='r', linestyle='--', linewidth=2)
            axs[0].axvline(x=self._times[self.linear_fit_window[1]], color='r', linestyle='--', linewidth=2)
        if plot_type == "log-log":
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
        elif plot_type == "linear-linear":
            axs[0].set_xscale("linear")
            axs[0].set_yscale("linear")

        axs[1].plot(
            self._times,
            self._lij_cation_anion,
            label="+ -",
            color="tab:orange"
        )
        i = int(len(self._lij_cation_anion) / 5)
        slope_guess = (self._lij_cation_anion[i] - self._lij_cation_anion[5]) / (self._times[i] - self._times[5])
        axs[1].plot(self._times, self._times * slope_guess * 2, "k--", alpha=0.5, label = "Slope = 1")  
        axs[1].set_ylabel("MSD +- (S*s/m)")
        axs[1].set_xlabel("Time (ps)")
        axs[1].set_title("Cation-Anion")
        axs[1].grid()
        axs[1].legend()
        if self.linear_fit_window:
            axs[1].axvline(x=self._times[self.linear_fit_window[0]], color='r', linestyle='--', linewidth=2)
            axs[1].axvline(x=self._times[self.linear_fit_window[1]], color='r', linestyle='--', linewidth=2)
        if plot_type == "log-log":
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
        elif plot_type == "linear-linear":
            axs[0].set_xscale("linear")
            axs[0].set_yscale("linear")

        axs[2].plot(
            self._times,
            self._lij_anion_anion,
            label="- -",
            color="tab:green"
        )
        i = int(len(self._lij_anion_anion) / 5)
        slope_guess = (self._lij_anion_anion[i] - self._lij_anion_anion[5]) / (self._times[i] - self._times[5])
        axs[2].plot(self._times, self._times * slope_guess * 2, "k--", alpha=0.5, label = "Slope = 1")  
        axs[2].set_ylabel("MSD -- (S*s/m)")
        axs[2].set_xlabel("Time (ps)")
        axs[2].set_title("Anion-Anion")
        axs[2].grid()
        axs[2].legend()
        if self.linear_fit_window:
            axs[2].axvline(x=self._times[self.linear_fit_window[0]], color='r', linestyle='--', linewidth=2)
            axs[2].axvline(x=self._times[self.linear_fit_window[1]], color='r', linestyle='--', linewidth=2)
        if plot_type == "log-log":
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
        elif plot_type == "linear-linear":
            axs[0].set_xscale("linear")
            axs[0].set_yscale("linear")
            
        plt.tight_layout()
        plt.show()
