from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from transport_analysis.utils import cross_corr, average_directions

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup

## Need to pass the entire the universe to get the com of the universe not just the atomg
class ConductivityKubo(AnalysisBase):
	"""
	Class to calculate the diffusion_coefficient of a species.
	Note that the slope of the mean square displacement provides
	the diffusion coeffiecient. This implementation uses FFT to compute MSD faster
	as explained here: https://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft
	Since we are using FFT, the sampling frequency (or) dump frequency of the simulation trajectory
	plays a critical role in the accuracy of the result.
	
	Particle velocities are required to calculate the viscosity function. Thus
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
        Dimensionality :math:`d` of the viscosity computation.
    results.timeseries : :class:`numpy.ndarray`
        The averaged viscosity function over all the particles with respect
        to lag-time. Obtained after calling :meth:`ViscosityHelfand.run`
    results.visc_by_particle : :class:`numpy.ndarray`
        The viscosity function of each individual particle with respect to
        lag-time.
    results.viscosity : float
        The viscosity coefficient of the solvent. The argument
        `linear_fit_window` must be provided to calculate this to
        avoid misinterpretation of the viscosity function.
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
	def __init__(self,
        allatomgroup: "AtomGroup",
		cationgroup_query: str,
		aniongroup_query: str,
        temp_avg: float = 298.0,
		cutoff_step: int = 10000,
		cation_num_atoms_per_species = int,
		anion_num_atoms_per_species = int, 
		cation_mass_per_species = float,
		anion_mass_per_species = float,
		cation_charge = int,
		anion_charge = int,
        **kwargs,
    ) -> None:
        # the below line must be kept to initialize the AnalysisBase class!
		super().__init__(allatomgroup.universe.trajectory, **kwargs)
        # after this you will be able to access `self.results`
        # `self.results` is a dictionary-like object
        # that can should used to store and retrieve results
        # See more at the MDAnalysis documentation:
        # https://docs.mdanalysis.org/stable/documentation_pages/analysis/base.html?highlight=results#MDAnalysis.analysis.base.Results
		if isinstance(allatomgroup, UpdatingAtomGroup):
			raise TypeError(
                "UpdatingAtomGroups are not valid for diffusion computation"
            )
		self.temp_avg = temp_avg
		self.cutoff_step = cutoff_step

		self.allatomgroup = allatomgroup
		self.cations = self.allatomgroup.select_atoms(cationgroup_query)
		self.anions = self.allatomgroup.select_atoms(aniongroup_query)
		self.cation_num_atoms_per_species = cation_num_atoms_per_species
		self.anion_num_atoms_per_species = anion_num_atoms_per_species
		self.cation_mass_per_species = cation_mass_per_species
		self.anion_mass_per_species = anion_mass_per_species
		self.cation_charge = cation_charge
		self.anion_charge = anion_charge

		self._dim = 3
		self.cation_particles = len(self.cations)
		self.anion_particles = len(self.anions)

		if (self.cation_particles%self.cation_num_atoms_per_species != 0) or (self.anion_particles%self.anion_num_atoms_per_species != 0):
			raise TypeError(
				"Some species are fragmented. Invalid AtomGroup"
			)

	def _prepare(self):
		"""
        Set up viscosity, mass, velocity, and position arrays
        before the analysis loop begins
        """
        # 2D viscosity array of frames x particles

		self._volumes = np.zeros((self.n_frames))

		self._acf_cation_cation = np.zeros((self.n_frames,self._dim))
		self._acf_cation_anion =  np.zeros((self.n_frames,self._dim))
		self._acf_anion_anion = np.zeros((self.n_frames,self._dim))

		self._l_cation_cation = np.zeros((self.n_frames-1, self._dim))
		self._l_cation_anion = np.zeros((self.n_frames-1, self._dim))
		self._l_anion_anion = np.zeros((self.n_frames-1, self._dim))
		self._lij_cation_cation = np.zeros((self.n_frames-1))
		self._lij_cation_anion = np.zeros((self.n_frames-1))
		self._lij_anion_anion = np.zeros((self.n_frames-1))

	
		self._cond = np.zeros((self.n_frames-1))
 
		self._coms_velocity = np.zeros((self.n_frames, self._dim))
		self._times = np.zeros((self.n_frames))
        
		self._cation_masses = self.cations.masses
		self._anion_masses = self.anions.masses
		self._all_masses = self.allatomgroup.masses

        # 3D arrays of frames x particles x dimensions
        # positions
		self._cation_velocties = np.zeros(
            (self.n_frames, int(self.cation_particles), self._dim)
        )
		self._anion_velocities = np.zeros(
            (self.n_frames, int(self.anion_particles), self._dim)
        )

		self._cation_mass_weighted_velocities = np.zeros(
            (self.n_frames, int(self.cation_particles/self.cation_num_atoms_per_species), self._dim)
        )

		self._anion_mass_weighted_velocities = np.zeros(
            (self.n_frames, int(self.anion_particles/self.anion_num_atoms_per_species), self._dim)
        )
		self.J_to_KJ = 1000
		self.boltzmann = self.J_to_KJ*constants["Boltzmann_constant"]/constants["N_Avogadro"]
		self.A_to_m = 1e-10
		self.fs_to_s = 1e-15
		self.e_to_C = constants['elementary_charge']
	
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
            and self._ts.volume != 0
        ):
			raise NoDataError(
                "Einstein diffusion computation requires positions, and box volume in the trajectory"
            )

		# fill volume array
		self._volumes[self._frame_index] = self._ts.volume
		self._times[self._frame_index] = self._ts.time
		# set shape of position array
		
		self._cation_velocties[self._frame_index] = self.cations.velocities
		self._anion_velocities[self._frame_index] = self.anions.velocities
		self._coms_velocity[self._frame_index] = np.multiply(np.repeat(self._all_masses,3,axis = 0).reshape(-1,3), self.allatomgroup.velocities).sum(axis = 0)/self._all_masses.sum()

		self._cation_mass_weighted_velocities[self._frame_index] = (np.multiply(np.repeat(self._cation_masses,self._dim,axis = 0).reshape(-1,self._dim),self._cation_velocties[self._frame_index])/self.cation_mass_per_species).reshape(-1, int(self.cation_num_atoms_per_species), self._dim).sum(axis=1)
		self._anion_mass_weighted_velocities[self._frame_index] = (np.multiply(np.repeat(self._anion_masses,self._dim,axis = 0).reshape(-1,self._dim),self._anion_velocities[self._frame_index])/self.anion_mass_per_species).reshape(-1, int(self.anion_num_atoms_per_species), self._dim).sum(axis=1)
		
		self._cation_mass_weighted_velocities[self._frame_index] = self._cation_mass_weighted_velocities[self._frame_index] - self._coms_velocity[self._frame_index] 
		self._anion_mass_weighted_velocities[self._frame_index] = self._anion_mass_weighted_velocities[self._frame_index] - self._coms_velocity[self._frame_index] 




	def _conclude(self):
		"""
		Calculates the conductivity coefficient via the fft.
		"""
		cation_summed_velocities = np.sum(self._cation_mass_weighted_velocities, axis=1)
		anion_summed_velocities = np.sum(self._anion_mass_weighted_velocities, axis=1)
		for i in range(self._dim):
			self._acf_cation_cation[:,i] = cross_corr(cation_summed_velocities[:,i],cation_summed_velocities[:,i])
			self._acf_anion_anion[:,i] = cross_corr(anion_summed_velocities[:,i],anion_summed_velocities[:,i])
			self._acf_cation_anion[:,i] = cross_corr(cation_summed_velocities[:,i],anion_summed_velocities[:,i])
		for i in range(self._dim):
			self._l_cation_cation[:,i] = integrate.cumulative_trapezoid(self._acf_cation_cation[:,i],self._times[:,i])
			self._l_anion_anion[:,i] = integrate.cumulative_trapezoid(self._acf_anion_anion[:,i],self._times[:,i])
			self._l_cation_anion[:,i] = integrate.cumulative_trapezoid(self._acf_cation_anion[:,i],self.times[:,i])
			
    
		self._lij_cation_cation = average_directions(self._l_cation_cation,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)
		self._lij_cation_anion = average_directions(self._l_anion_anion,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)
		self._lij_anion_anion = average_directions(self._l_cation_anion,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)

		self._cond = (self.cation_charge**2)*self._lij_cation_cation + (self.anion_charge**2)*self._lij_anion_anion + (2*self.cation_charge*self.anion_charge)*self._lij_cation_anion

		self.results.conductivity = self._cond[self.cutoff_step]*(self.e_to_C*self.e_to_C)*(1/(self.fs_to_s*self.A_to_m))*10
				

	def plot_acf(self):
		"""
		Plot the mead square displacement vs time and the fit region in the log-log scale
		"""
		plt.plot(self._times[:self.cutoff_step],self._acf_cation_cation[:self.cutoff_step], label = '+ +')
		plt.plot(self._times[:self.cutoff_step],self._acf_cation_anion[:self.cutoff_step], label = '+ -')
		plt.plot(self._times[:self.cutoff_step],self._acf_anion_anion[:self.cutoff_step], label = '- -')

		plt.grid()
		plt.ylabel("Velocity Autocorrelation Function")
		plt.xlabel("Time")
		plt.tight_layout()
		plt.show()

	
		





		




