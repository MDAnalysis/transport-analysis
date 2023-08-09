"""
Viscosity --- :mod:`transport_analysis.analysis.viscosity`
==========================================================

This module offers a class for the lightweight computation of
viscosity via the Einstein-Helfand method.

"""
from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants
import numpy as np

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup


class ViscosityHelfand(AnalysisBase):
    """
    Class to calculate viscosity using the Einstein-Helfand approach.

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

    Attributes
    ----------
    atomgroup : :class:`~MDAnalysis.core.groups.AtomGroup`
        The atoms to which this analysis is applied
    dim_fac : int
        Dimensionality :math:`d` of the viscosity computation.
    results.timeseries : :class:`numpy.ndarray`
        The averaged viscosity over all the particles with respect to lag-time.
        Obtained after calling :meth:`ViscosityHelfand.run`
    results.visc_by_particle : :class:`numpy.ndarray`
        The viscosity of each individual particle with respect to lag-time.
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
        atomgroup: "AtomGroup",
        temp_avg: float = 300.0,
        dim_type: str = "xyz",
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
                "UpdatingAtomGroups are not valid for viscosity computation"
            )

        # args
        self.temp_avg = temp_avg
        self.dim_type = dim_type.lower()
        self._dim, self.dim_fac = self._parse_dim_type(self.dim_type)
        # self.fft = fft # consider whether fft is possible later

        # local
        self.atomgroup = atomgroup
        self.n_particles = len(self.atomgroup)

    def _prepare(self):
        """
        Set up viscosity, mass, velocity, and position arrays
        before the analysis loop begins
        """
        # 2D viscosity array of frames x particles
        self.results.visc_by_particle = np.zeros(
            (self.n_frames, self.n_particles)
        )

        self._volumes = np.zeros((self.n_frames))

        self._masses = self.atomgroup.masses
        self._masses_rs = self._masses.reshape((1, len(self._masses), 1))

        # 3D arrays of frames x particles x dimensionality
        # for velocities and positions
        self._velocities = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac)
        )

        self._positions = np.zeros(
            (self.n_frames, self.n_particles, self.dim_fac)
        )
        # self.results.timeseries not set here

        # update when mda 2.6.0 releases with typo fix
        try:
            self.boltzmann = constants["Boltzmann_constant"]
        except KeyError:
            self.boltzmann = constants["Boltzman_constant"]

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

    def _single_frame(self):
        """
        Constructs arrays of velocities and positions
        for viscosity computation.
        """
        # This runs once for each frame of the trajectory

        # The trajectory positions update automatically
        # You can access the frame number using self._frame_index

        # trajectory must have velocity and position information
        if not (
            self._ts.has_velocities
            and self._ts.has_positions
            and self._ts.volume != 0
        ):
            raise NoDataError(
                "Helfand viscosity computation requires "
                "velocities, positions, and box volume in the trajectory"
            )

        # fill volume array
        self._volumes[self._frame_index] = self._ts.volume

        # set shape of velocity array
        self._velocities[self._frame_index] = self.atomgroup.velocities[
            :, self._dim
        ]

        # set shape of position array
        self._positions[self._frame_index] = self.atomgroup.positions[
            :, self._dim
        ]

    def _conclude(self):
        r"""Calculates viscosity via the simple "windowed" algorithm."""
        self._vol_avg = np.average(self._volumes)

        lagtimes = np.arange(1, self.n_frames)

        # iterate through all possible lagtimes from 1 to number of frames
        for lag in lagtimes:
            # get difference of momentum * position shifted by "lag" frames
            diff = (
                self._masses_rs
                * self._velocities[:-lag, :, :]
                * self._positions[:-lag, :, :]
                - self._masses_rs
                * self._velocities[lag:, :, :]
                * self._positions[lag:, :, :]
            )

            # square and sum each x(, y, z) diff per particle
            sq_diff = np.square(diff).sum(axis=-1)

            # average over # frames
            # update viscosity by particle array
            self.results.visc_by_particle[lag, :] = np.mean(sq_diff, axis=0)

        # divide by 2, boltzmann constant, vol_avg, and temp_avg
        self.results.visc_by_particle = self.results.visc_by_particle / (
            2 * self.boltzmann * self._vol_avg * self.temp_avg
        )
        # average over # particles and update results array
        self.results.timeseries = self.results.visc_by_particle.mean(axis=1)
