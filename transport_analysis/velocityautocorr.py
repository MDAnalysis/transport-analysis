"""
Velocity Autocorrelation Function --- :mod:`transport_analysis.velocityautocorr`
================================================================================

This module offers a class to efficiently calculate a velocity
autocorrelation function (VACF). Averaging over all atoms in the atom group
``ag``, regardless of type, it will calculate

.. math::
    C(j \Delta t) = {1 \over N - j} \sum_{i=0}^{N-1-j}
    v(i \Delta t)v((i+j)\Delta t)

where :math:`N` is the number of time frames and :math:`\Delta t` are
discrete time intervals between data points.

Basic usage
-----------

This example uses the files :data:`~MDAnalysis.tests.datafiles.PRM_NCBOX` and
:data:`~MDAnalysis.tests.datafiles.TRJ_NCBOX` from the MDAnalysis test suite.
To get started, execute  ::

   # imports
   >>> import MDAnalysis as mda
   >>> from transport_analysis.velocityautocorr import VelocityAutocorr

   # test data for this example
   >>> from MDAnalysis.tests.datafiles import PRM_NCBOX, TRJ_NCBOX

We will calculate the VACF of an atom group of all water atoms in
residues 1-5. To select these atoms:

   >>> u = mda.Universe(PRM_NCBOX, TRJ_NCBOX)
   >>> ag = u.select_atoms("resname WAT and resid 1-5")

We can run the calculation using any variable of choice such as
``wat_vacf`` and access our results using ``wat_vacf.results.timeseries``:

   >>> wat_vacf = VelocityAutocorr(ag).run()
   >>> wat_vacf.results.timeseries
   array([275.62075467, -18.42008255, -23.94383428,  41.41415381,
        -2.3164344 , -35.66393559, -22.66874897,  -3.97575003,
         6.57888933,  -5.29065096])

Notice that this example data is insufficient to provide a well-defined VACF.
When working with real data, ensure that the frames are captured frequently
enough to obtain a VACF suitable for your needs.

"""
from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
import numpy as np
import tidynamics
import matplotlib.pyplot as plt
from scipy import integrate
from transport_analysis.due import due, Doi

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup

due.cite(
    Doi("10.21105/joss.00877"),
    description="Autocorrelation with tidynamics",
    path="transport_analysis.velocityautocorr",
    cite_module=True,
)


class VelocityAutocorr(AnalysisBase):
    """
    Class to calculate a velocity autocorrelation function (VACF).

    Parameters
    ----------
    atomgroup : AtomGroup
        An MDAnalysis :class:`~MDAnalysis.core.groups.AtomGroup`.
        Note that :class:`UpdatingAtomGroup` instances are not accepted.
    dim_type : {'xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'}
        Desired dimensions to be included in the VACF. Defaults to 'xyz'.
    fft : bool
        If ``True``, uses a fast FFT based algorithm for computation of
        the VACF. Otherwise, use the simple "windowed" algorithm.
        The tidynamics package is required for `fft=True`.
        Defaults to ``True``.

    Attributes
    ----------
    atomgroup : :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    dim_fac : int
        Dimensionality :math:`d` of the VACF.
    results.timeseries : :class:`numpy.ndarray`
        The averaged VACF over all the particles with respect to lag-time.
        Obtained after calling :meth:`VelocityAutocorr.run`
    results.vacf_by_particle : :class:`numpy.ndarray`
        The VACF of each individual particle with respect to lag-time.
    start : Optional[int]
        The first frame of the trajectory used to compute the analysis.
    stop : Optional[int]
        The frame to stop at for the analysis, non-inclusive.
    step : Optional[int]
        Number of frames to skip between each analyzed frame.
    n_frames : int
        Number of frames analysed in the trajectory.
    n_particles : int
        Number of particles VACF was calculated over.
    """

    def __init__(
        self,
        atomgroup: "AtomGroup",
        dim_type: str = "xyz",
        fft: bool = True,
        **kwargs,
    ) -> None:
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results

        if isinstance(atomgroup, UpdatingAtomGroup):
            raise TypeError(
                "UpdatingAtomGroups are not valid for VACF computation"
            )

        # args
        self.dim_type = dim_type.lower()
        self._dim, self.dim_fac = self._parse_dim_type(self.dim_type)
        self.fft = fft

        # local
        self.atomgroup = atomgroup
        self.n_particles = len(self.atomgroup)
        self._run_called = False

    def _prepare(self):
        """Set up velocity and VACF arrays before the analysis loop begins"""
        # 2D array of frames x particles
        self.results.vacf_by_particle = np.zeros(
            (self.n_frames, self.n_particles)
        )

        # 3D array of frames x particles x dimensionality
        self._velocities = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac)
        )
        # self.results.timeseries not set here

    @staticmethod
    def _parse_dim_type(dim_str):
        """Sets up the desired dimensionality of the VACF."""
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

    def _single_frame(self):
        """Constructs array of velocities for VACF calculation."""
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # trajectory must have velocity information
        if not self._ts.has_velocities:
            raise NoDataError(
                "VACF computation requires velocities in the trajectory"
            )

        # set shape of velocity array
        self._velocities[self._frame_index] = self.atomgroup.velocities[
            :, self._dim
        ]

    # Results will be in units of (angstroms / ps)^2
    def _conclude(self):
        """Calculate the final results of the analysis"""
        # This is an optional method that runs after
        # _single_frame loops over the trajectory.
        # It is useful for calculating the final results
        # of the analysis.
        if self.fft:
            self._conclude_fft()
        else:
            self._conclude_simple()

    def _conclude_fft(self):  # with FFT, np.float64 bit prescision required.
        r"""Calculates the VACF via the FCA fast correlation algorithm."""
        for n in range(self.n_particles):
            self.results.vacf_by_particle[:, n] = tidynamics.acf(
                self._velocities[:, n, :]
            )
        self.results.timeseries = self.results.vacf_by_particle.mean(axis=1)
        self._run_called = True

    def _conclude_simple(self):
        r"""Calculates the VACF via the simple "windowed" algorithm."""
        # total frames in trajectory, use N for readability
        N = self.n_frames

        # iterate through all possible lagtimes up to N
        for lag in range(N):
            # get product of velocities shifted by "lag" frames
            veloc = (
                self._velocities[: N - lag, :, :]
                * self._velocities[lag:, :, :]
            )

            # dot product of x(, y, z) velocities per particle
            sum_veloc = np.sum(veloc, axis=-1)

            # average over # frames
            # update VACF by particle array
            self.results.vacf_by_particle[lag, :] = np.mean(sum_veloc, axis=0)
        # average over # particles and update results array
        self.results.timeseries = self.results.vacf_by_particle.mean(axis=1)
        self._run_called = True

    def plot_vacf(
        self,
        start=0,
        stop=0,
        step=1,
        xlabel="Time (ps)",
        ylabel="Velocity Autocorrelation Function (Å^2 / ps^2)",
    ):
        """
        Returns a velocity autocorrelation function (VACF) plot via
        ``Matplotlib``. Analysis must be run prior to plotting.

        Parameters
        ----------
        start : Optional[int]
            The first frame of ``self.results.timeseries``
            used for the plot.
        stop : Optional[int]
            The frame of ``self.results.timeseries`` to stop at
            for the plot, non-inclusive.
        step : Optional[int]
            Number of frames to skip between each plotted frame.
        xlabel : Optional[str]
            The x-axis label text. Defaults to "Time (ps)".
        ylabel : Optional[str]
            The y-axis label text.
            Defaults to "Velocity Autocorrelation Function (Å^2 / ps^2)".

        Returns
        -------
        :class:`matplotlib.lines.Line2D`
            A :class:`matplotlib.lines.Line2D` instance with
            the desired VACF plotting information.
        """
        if not self._run_called:
            raise RuntimeError("Analysis must be run prior to plotting")

        stop = self.n_frames if stop == 0 else stop

        fig, ax_vacf = plt.subplots()
        ax_vacf.set_xlabel(xlabel)
        ax_vacf.set_ylabel(ylabel)
        return ax_vacf.plot(
            self.times[start:stop:step],
            self.results.timeseries[start:stop:step],
        )

    def self_diffusivity_gk(self, start=0, stop=0, step=1):
        """
        Returns a self-diffusivity value using ``scipy.integrate.trapezoid``.
        Analysis must be run prior to computing self-diffusivity.

        Parameters
        ----------
        start : Optional[int]
            The first frame of ``self.results.timeseries``
            used for the calculation.
        stop : Optional[int]
            The frame of ``self.results.timeseries`` to stop at
            for the calculation, non-inclusive.
        step : Optional[int]
            Number of frames to skip between each frame used
            for the calculation.

        Returns
        -------
        `numpy.float64`
            The calculated self-diffusivity value for the analysis.
        """
        if not self._run_called:
            raise RuntimeError(
                "Analysis must be run prior to computing self-diffusivity"
            )

        stop = self.n_frames if stop == 0 else stop

        return (
            integrate.trapezoid(
                self.results.timeseries[start:stop:step],
                self.times[start:stop:step],
            )
            / self.dim_fac
        )

    def self_diffusivity_gk_odd(self, start=0, stop=0, step=1):
        """
        Returns a self-diffusivity value using ``scipy.integrate.simpson``.
        Recommended for use with an odd number of evenly spaced data points.
        Analysis must be run prior to computing self-diffusivity.

        Parameters
        ----------
        start : Optional[int]
            The first frame of ``self.results.timeseries``
            used for the calculation.
        stop : Optional[int]
            The frame of ``self.results.timeseries`` to stop at
            for the calculation, non-inclusive.
        step : Optional[int]
            Number of frames to skip between each frame used
            for the calculation.

        Returns
        -------
        `numpy.float64`
            The calculated self-diffusivity value for the analysis.
        """
        if not self._run_called:
            raise RuntimeError(
                "Analysis must be run prior to computing self-diffusivity"
            )

        stop = self.n_frames if stop == 0 else stop

        return (
            integrate.simpson(
                y=self.results.timeseries[start:stop:step],
                x=self.times[start:stop:step],
            )
            / self.dim_fac
        )

    def plot_running_integral(
        self,
        start=0,
        stop=0,
        step=1,
        initial=0,
        xlabel="Time (ps)",
        ylabel="Running Integral of the VACF (Å^2 / ps)",
    ):
        """
        Returns a plot of the running integral of the
        velocity autocorrelation function (VACF) via ``Matplotlib``.
        In this case, the running integral is the integral of the VACF
        divided by the dimensionality. Analysis must be run prior to plotting.

        Parameters
        ----------
        start : Optional[int]
            The first frame of ``self.results.timeseries``
            used for the plot.
        stop : Optional[int]
            The frame of ``self.results.timeseries`` to stop at
            for the plot, non-inclusive.
        step : Optional[int]
            Number of frames to skip between each plotted frame.
        initial : Optional[float]
            Inserted value at the beginning of the integrated result array.
            Defaults to 0.
        xlabel : Optional[str]
            The x-axis label text. Defaults to "Time (ps)".
        ylabel : Optional[str]
            The y-axis label text.
            Defaults to "Running Integral of the VACF (Å^2 / ps)".

        Returns
        -------
        :class:`matplotlib.lines.Line2D`
            A :class:`matplotlib.lines.Line2D` instance with
            the desired VACF plotting information.
        """
        if not self._run_called:
            raise RuntimeError("Analysis must be run prior to plotting")

        stop = self.n_frames if stop == 0 else stop

        running_integral = (
            integrate.cumulative_trapezoid(
                self.results.timeseries[start:stop:step],
                self.times[start:stop:step],
                initial=initial,
            )
            / self.dim_fac
        )

        fig, ax_running_integral = plt.subplots()
        ax_running_integral.set_xlabel(xlabel)
        ax_running_integral.set_ylabel(ylabel)
        return ax_running_integral.plot(
            self.times[start:stop:step],
            running_integral,
        )
