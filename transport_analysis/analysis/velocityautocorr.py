"""
VelocityAutocorr --- :mod:`transport_analysis.analysis.VelocityAutocorr`
===========================================================

This module contains the :class:`VelocityAutocorr` class.

"""
from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
import numpy as np

if TYPE_CHECKING:
    from MDAnalysis.core.universe import Universe, AtomGroup


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
        dim_type="xyz",
        # fft=True,
        **kwargs
    ):
        # the below line must be kept to initialize the AnalysisBase class!
        super().__init__(atomgroup.universe.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results

        # args
        self.dim_type = dim_type
        # self._parse_dim_type()
        # self.fft = fft

        # local
        self.atomgroup = atomgroup
        self.n_particles = len(self.atomgroup)
        self._velocity_array = None

        # result
        self.results.vacf_by_particle = None
        self.results.timeseries = None

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        # This is an optional method that runs before
        # _single_frame loops over the trajectory.
        # It is useful for setting up results arrays
        # For example, below we create an array to store
        # the number of atoms with negative coordinates
        # in each frame.
        self.results.is_negative = np.zeros(
            (self.n_frames, self.atomgroup.n_atoms),
            dtype=bool,
        )

    def _single_frame(self):
        """Calculate data from a single frame of trajectory"""
        # This runs once for each frame of the trajectory
        # It can contain the main analysis method, or just collect data
        # so that analysis can be done over the aggregate data
        # in _conclude.

        # The trajectory positions update automatically
        negative = self.atomgroup.positions < 0
        # You can access the frame number using self._frame_index
        self.results.is_negative[self._frame_index] = negative.any(axis=1)

    def _conclude(self):
        """Calculate the final results of the analysis"""
        # This is an optional method that runs after
        # _single_frame loops over the trajectory.
        # It is useful for calculating the final results
        # of the analysis.
        # For example, below we determine the
        # which atoms always have negative coordinates.
        self.results.always_negative = self.results.is_negative.all(axis=0)
        always_negative_atoms = self.atomgroup[self.results.always_negative]
        self.results.always_negative_atoms = always_negative_atoms
        self.results.always_negative_atom_names = always_negative_atoms.names

        # results don't have to be arrays -- they can be any value, e.g. floats
        self.results.n_negative_atoms = self.results.is_negative.sum(axis=1)
        self.results.mean_negative_atoms = self.results.n_negative_atoms.mean()
