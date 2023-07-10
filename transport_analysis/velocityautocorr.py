"""
VelocityAutocorr --- :mod:`transport_analysis.analysis.VelocityAutocorr`
========================================================================

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

   >>> import transport_analysis as ta
   >>> from MDAnalysis.tests.datafiles import PRM_NCBOX, TRJ_NCBOX

We will calculate the VACF of an atom group of all water atoms in
residues 1-5. To select these atoms:

   >>> u = mda.Universe(PRM_NCBOX, TRJ_NCBOX)
   >>> ag = u.select_atoms("resname WAT and resid 1-5")

We can run the calculation using any variable of choice such as
``wat_vacf`` and access our results using ``wat_vacf.results.timeseries``:

   >>> wat_vacf = ta.VelocityAutocorr(ag).run()
   >>> wat_vacf.results.timeseries
   [275.62075467 -18.42008255 -23.94383428  41.41415381  -2.3164344
   -35.66393559 -22.66874897  -3.97575003   6.57888933  -5.29065096]

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
from matplotlib import pyplot as plt
from transport_analysis.due import due, Doi

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup

due.cite(
    Doi("10.21105/joss.00877"),
    description="Autocorrelation with tidynamics",
    path="transport_analysis.analysis.velocityautocorr",
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
        The frame to stop at for the analysis.
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
        self.dim_type = dim_type
        self._parse_dim_type()
        self.fft = fft

        # local
        self.atomgroup = atomgroup
        self.n_particles = len(self.atomgroup)

    def _prepare(self):
        """Set up velocity and VACF arrays before the analysis loop begins"""
        # 2D array of frames x particles
        self.results.vacf_by_particle = np.zeros(
            (self.n_frames, self.n_particles)
        )

        # 3D array of frames x particles x dimensionality
        self._velocity_array = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac)
        )
        # self.results.timeseries not set here

    def _parse_dim_type(self):
        r"""Sets up the desired dimensionality of the VACF."""
        keys = {
            "x": [0],
            "y": [1],
            "z": [2],
            "xy": [0, 1],
            "xz": [0, 2],
            "yz": [1, 2],
            "xyz": [0, 1, 2],
        }

        self.dim_type = self.dim_type.lower()

        try:
            self._dim = keys[self.dim_type]
        except KeyError:
            raise ValueError(
                "invalid dim_type: {} specified, please specify one of xyz, "
                "xy, xz, yz, x, y, z".format(self.dim_type)
            )

        self.dim_fac = len(self._dim)

    def _single_frame(self):
        """Constructs array of velocities for VACF calculation."""
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # trajectory must have velocity information
        if not self._ts.has_velocities:
            raise NoDataError(
                "VACF computation requires velocities " "in the trajectory"
            )

        # set shape of velocity array
        self._velocity_array[self._frame_index] = self.atomgroup.velocities[
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
        velocities = self._velocity_array.astype(np.float64)
        for n in range(self.n_particles):
            self.results.vacf_by_particle[:, n] = tidynamics.acf(
                velocities[:, n, :]
            )
        self.results.timeseries = self.results.vacf_by_particle.mean(axis=1)

    def _conclude_simple(self):
        r"""Calculates the VACF via the simple "windowed" algorithm."""
        # total frames in trajectory, use N for readability
        N = self.n_frames

        # improve precision with np.float64
        velocities = self._velocity_array.astype(np.float64)

        # iterate through all possible lagtimes up to N
        for lag in range(N):
            # get product of velocities shifted by "lag" frames
            veloc = velocities[: N - lag, :, :] * velocities[lag:, :, :]

            # dot product of x(, y, z) velocities per particle
            sum_veloc = np.sum(veloc, axis=-1)

            # average over # frames
            # update VACF by particle array
            self.results.vacf_by_particle[lag, :] = np.mean(sum_veloc, axis=0)
        # average over # particles and update results array
        self.results.timeseries = self.results.vacf_by_particle.mean(axis=1)


def plot_vacf(vacf_obj, start=0, stop=0, step=1):
    stop = vacf_obj.n_frames if stop == 0 else stop

    plt.xlabel('Time (ps)')
    plt.ylabel('Velocity Autocorrelation Function (VACF) (Å)')
    plt.title('Velocity Autocorrelation Function vs. Time')
    plt.plot(vacf_obj.times[start:stop:step],
             vacf_obj.results.timeseries[start:stop:step])
