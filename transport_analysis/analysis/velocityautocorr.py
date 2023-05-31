"""
VelocityAutocorr --- :mod:`transport_analysis.analysis.VelocityAutocorr`
===========================================================

This module contains the :class:`VelocityAutocorr` class.

"""
from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
import numpy as np

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup


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
    times : numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`VelocityAutocorr.run`
    frames : numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`VelocityAutocorr.run`
    """

    def __init__(
        self,
        atomgroup: "AtomGroup",
        dim_type: str = "xyz",
        fft: bool = True,
        **kwargs
    ) -> None:
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results

        try:
            if not atomgroup.universe.trajectory.has_velocities:
                raise AttributeError
        except AttributeError:
            raise AttributeError("atomgroup must be from a trajectory with velocities")
        else:
            if isinstance(atomgroup, UpdatingAtomGroup):
                raise TypeError("UpdatingAtomGroups are not valid for VACF "
                                "computation")

            # args
            self.dim_type = dim_type
            self._parse_dim_type()
            self.fft = fft

            # local
            self.atomgroup = atomgroup
            self.n_particles = len(self.atomgroup)
            self._velocity_array = None

            # result
            self.results.vacf_by_particle = None
            self.results.timeseries = None

    def _prepare(self):
        """Set up velocity and VACF arrays before the analysis loop begins"""
        # 2D array of frames x particles
        self.results.vacf_by_particle = np.zeros((self.n_frames,
                                                  self.n_particles))

        # 3D array of frames x particles x dimensionality
        self._velocity_array = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac))
        # self.results.timeseries not set here

    def _parse_dim_type(self):
        r""" Sets up the desired dimensionality of the VACF.

        """
        keys = {'x': [0], 'y': [1], 'z': [2], 'xy': [0, 1],
                'xz': [0, 2], 'yz': [1, 2], 'xyz': [0, 1, 2]}

        self.dim_type = self.dim_type.lower()

        try:
            self._dim = keys[self.dim_type]
        except KeyError:
            raise ValueError(
                'invalid dim_type: {} specified, please specify one of xyz, '
                'xy, xz, yz, x, y, z'.format(self.dim_type))

        self.dim_fac = len(self._dim)

    def _single_frame(self):
        """Constructs array of velocities for VACF calculation."""
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # set shape of velocity array
        # trajectory must have velocity information
        self._velocity_array[self._frame_index] = (
            self.atomgroup.velocities[:, self._dim])

    def _conclude(self):
        """Calculate the final results of the analysis"""
        # This is an optional method that runs after
        # _single_frame loops over the trajectory.
        # It is useful for calculating the final results
        # of the analysis.
        pass
