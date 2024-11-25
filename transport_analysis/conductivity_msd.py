from typing import TYPE_CHECKING

from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import UpdatingAtomGroup
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.units import constants
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transport_analysis.utils import msd_fft_cross_1d, msd_variance_cross_1d, average_directions

if TYPE_CHECKING:
    from MDAnalysis.core.universe import AtomGroup

## Need to pass the entire the universe to get the com of the universe not just the atomg
class ConductivityEinstein(AnalysisBase):
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
        linear_fit_window: tuple[int, int] = None,
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
		self.linear_fit_window = linear_fit_window

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

		self._msds_cation_cation = np.zeros((self.n_frames,self._dim))
		self._msds_cation_anion =  np.zeros((self.n_frames,self._dim))
		self._msds_anion_anion = np.zeros((self.n_frames,self._dim))

		self._msds_cation_cation_var = np.zeros((self.n_frames,self._dim))
		self._msds_cation_anion_var = np.zeros((self.n_frames,self._dim))
		self._msds_anion_anion_var = np.zeros((self.n_frames,self._dim))
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
		self._cation_positions = np.zeros(
            (self.n_frames, int(self.cation_particles), self._dim)
        )
		self._anion_positions = np.zeros(
            (self.n_frames, int(self.anion_particles), self._dim)
        )

		self._cation_mass_weighted_positions = np.zeros(
            (self.n_frames, int(self.cation_particles/self.cation_num_atoms_per_species), self._dim)
        )

		self._anion_mass_weighted_positions = np.zeros(
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
            self._ts.has_positions
            and self._ts.volume != 0
        ):
			raise NoDataError(
                "Einstein diffusion computation requires positions, and box volume in the trajectory"
            )

		# fill volume array
		self._volumes[self._frame_index] = self._ts.volume
		self._times[self._frame_index] = self._ts.time
		# set shape of position array
		
		self._cation_positions[self._frame_index] = self.cations.positions
		self._anion_positions[self._frame_index] = self.anions.positions
		self._coms[self._frame_index] = self.allatomgroup.center_of_mass(wrap=False)

		self._cation_mass_weighted_positions[self._frame_index] = (np.multiply(np.repeat(self._cation_masses,self._dim,axis = 0).reshape(-1,self._dim),self._cation_positions[self._frame_index])/self.cation_mass_per_species).reshape(-1, int(self.cation_num_atoms_per_species), self._dim).sum(axis=1)
		self._anion_mass_weighted_positions[self._frame_index] = (np.multiply(np.repeat(self._anion_masses,self._dim,axis = 0).reshape(-1,self._dim),self._anion_positions[self._frame_index])/self.anion_mass_per_species).reshape(-1, int(self.anion_num_atoms_per_species), self._dim).sum(axis=1)
		
		self._cation_mass_weighted_positions[self._frame_index] = self._cation_mass_weighted_positions[self._frame_index] - self._coms[self._frame_index] 
		self._anion_mass_weighted_positions[self._frame_index] = self._anion_mass_weighted_positions[self._frame_index] - self._coms[self._frame_index] 




	def _conclude(self):
		"""
		Calculates the conductivity coefficient via the fft.
		"""
		cation_summed_positions = np.sum(self._cation_mass_weighted_positions, axis=1)
		anion_summed_positions = np.sum(self._anion_mass_weighted_positions, axis=1)

		self._msds_cation_cation = np.transpose(
        	[msd_fft_cross_1d(cation_summed_positions[:, i], cation_summed_positions[:, i]) for i in range(self._dim)]
    	)
		self._msds_cation_cation_var = np.transpose(
        	[msd_variance_cross_1d(cation_summed_positions[:, i], cation_summed_positions[:, i], self._msds_cation_cation[:, i]) for i in range(self._dim)]
    	)
		self._msds_anion_anion = np.transpose(
        	[msd_fft_cross_1d(anion_summed_positions[:, i], anion_summed_positions[:, i]) for i in range(self._dim)]
    	)
		self._msds_anion_anion_var = np.transpose(
        	[msd_variance_cross_1d(anion_summed_positions[:, i], anion_summed_positions[:, i], self._msds_anion_anion[:, i]) for i in range(self._dim)]
    	)
		self._msds_cation_anion = np.transpose(
        	[msd_fft_cross_1d(cation_summed_positions[:, i], anion_summed_positions[:, i]) for i in range(self._dim)]
    	)
		self._msds_cation_anion_var = np.transpose(
        	[msd_variance_cross_1d(cation_summed_positions[:, i], anion_summed_positions[:, i], self._msds_cation_anion[:, i]) for i in range(self._dim)]
    	)
		self._lij_cation_cation = average_directions(self._msds_cation_cation,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)
		self._lij_cation_anion = average_directions(self._msds_cation_anion,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)
		self._lij_anion_anion = average_directions(self._msds_anion_anion,self._dim)/(2*self.boltzmann*self.temp_avg*self._volumes)

		#self._cond = (self.cation_charge**2)*self._lij_cation_cation + (self.anion_charge**2)*self._lij_anion_anion + (2*self.cation_charge*self.anion_charge)*self._lij_cation_anion
		#self._cond_var =  ((self.cation_charge**2)**2)*self._msds_cation_cation_var + ((self.anion_charge**2)**2)*self._msds_anion_anion_var + ((2*self.cation_charge*self.anion_charge)**2)*self._msds_cation_anion_var
		
		self._lij_cation_cation_weights = np.sqrt(np.abs(1/average_directions(self._msds_cation_cation_var,self._dim)))
		self._lij_cation_cation_weights /= np.sum(self._lij_cation_cation_weights)

		self._lij_cation_anion_weights = np.sqrt(np.abs(1/average_directions(self._msds_cation_anion_var,self._dim)))
		self._lij_cation_anion_weights /= np.sum(self._lij_cation_anion_weights)

		self._lij_anion_anion_weights = np.sqrt(np.abs(1/average_directions(self._msds_anion_anion_var,self._dim)))
		self._lij_anion_anion_weights /= np.sum(self._lij_anion_anion_weights)

		cond = None
		lij_cation_cation = None
		lij_cation_anion = None
		lij_anion_anion = None
		beta = None
		if self.linear_fit_window:
			lij_cation_cation , _ , _ , _, _ = stats.linregress(self._times[self.linear_fit_window[0]:self.linear_fit_window[1]], self._lij_cation_cation[self.linear_fit_window[0]:self.linear_fit_window[1]])
			lij_cation_anion , _ , _ , _, _ = stats.linregress(self._times[self.linear_fit_window[0]:self.linear_fit_window[1]], self._lij_cation_anion[self.linear_fit_window[0]:self.linear_fit_window[1]])
			lij_anion_anion , _ , _ , _, _ = stats.linregress(self._times[self.linear_fit_window[0]:self.linear_fit_window[1]], self._lij_anion_anion[self.linear_fit_window[0]:self.linear_fit_window[1]])
			cond = (self.cation_charge**2)*lij_cation_cation + (self.anion_charge**2)*lij_anion_anion + (2*self.cation_charge*self.anion_charge)*lij_cation_anion
			
			self._cond = (self.cation_charge**2)*self._lij_cation_cation + (self.anion_charge**2)*self._lij_anion_anion + (2*self.cation_charge*self.anion_charge)*self._lij_cation_anion
			fit_slope = np.gradient(np.log(self._cond[self.linear_fit_window[0]:self.linear_fit_window[1]])[1:], np.log(self._times[self.linear_fit_window[0]:self.linear_fit_window[1]] - self._times[0])[1:])
			beta = np.nanmean(np.array(fit_slope))

		else:
			_ , lij_cation_cation = np.polynomial.polynomial.polyfit(self._times[1:], self._msds_cation_cation[1:], 1, w=self._lij_cation_cation_weights[1:])
			_ , lij_cation_anion = np.polynomial.polynomial.polyfit(self._times[1:], self._msds_cation_anion[1:], 1, w=self._lij_cation_anion_weights[1:])
			_ , lij_anion_anion = np.polynomial.polynomial.polyfit(self._times[1:], self._msds_anion_anion[1:], 1, w=self._lij_anion_anion_weights[1:])
			cond = (self.cation_charge**2)*lij_cation_cation + (self.anion_charge**2)*lij_anion_anion + (2*self.cation_charge*self.anion_charge)*lij_cation_anion

			#fit_slope = np.gradient(np.log(self._cond)[1:], np.log(self._times - self._times[0])[1:])
			#beta = np.average(np.array(fit_slope), weights=self.weights[1:])
		if self.linear_fit_window:
			self.results.linearity = beta
		#self.results.fit_slope = cond
		#self.results.fit_intercept = cond_intercept
		cation_mass_fraction = self._cation_masses.sum()/self.allatomgroup.masses.sum()
		anion_mass_fraction = self._anion_masses.sum()/self.allatomgroup.masses.sum()
		solvent_fraction = 1 - cation_mass_fraction - anion_mass_fraction

		self.results.conductivity = cond*(self.e_to_C*self.e_to_C)*(1/(self.fs_to_s*self.A_to_m))*10
		self.results.cation_transference_com = ((self.cation_charge**2)*lij_cation_cation + (self.cation_charge*self.anion_charge)*lij_cation_anion)/cond
		self.results.anion_transference_com = ((self.anion_charge**2)*lij_anion_anion + (self.cation_charge*self.anion_charge)*lij_cation_anion)/cond
		self.results.cation_transference_solvent = (self.results.cation_transference_com - anion_mass_fraction)/solvent_fraction
		self.results.anion_transference_solvent = (self.results.anion_transference_com -cation_mass_fraction)/solvent_fraction
		
	def plot_linear_fit(self):
		"""
		Plot the mead square displacement vs time and the fit region in the log-log scale
		"""
		plt.plot(self._times, np.abs(self._lii_self))
		plt.plot(self._times, self._times*self.results.fit_slope + self.results.fit_intercept, "k--", alpha=0.5)
		plt.grid()
		plt.ylabel("MSD")
		plt.xlabel("Time")
		if self.linear_fit_window:
			plt.axvline(x=self._times[self.linear_fit_window[0]], color='r', linestyle='--', linewidth=2)
			plt.axvline(x=self._times[self.linear_fit_window[1]], color='r', linestyle='--', linewidth=2)
		#plt.ylim(min(np.abs(self._msds[1:])) * 0.9, max(np.abs(self._msds)) * 1.1)
		#i = int(len(self._msds) / 5)
		#slope_guess = (self._msds[i] - self._msds[5]) / (self._times[i] - self._times[5])
		#plt.plot(self._times[self.linear_fit_window[0]:self.linear_fit_window[1]], self._times[self.linear_fit_window[0]:self.linear_fit_window[1]] * slope_guess * 2, "k--", alpha=0.5)
		plt.tight_layout()
		plt.show()

	
		





		




