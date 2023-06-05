import pytest
from numpy.testing import assert_almost_equal

from transport_analysis.analysis.velocityautocorr import (
    VelocityAutocorr as VACF
)
import MDAnalysis as mda
import numpy as np

from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import PRM_NCBOX, TRJ_NCBOX
from MDAnalysisTests.util import block_import, import_not_available


@pytest.fixture(scope='module')
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


@pytest.fixture(scope='module')
def ag(u):
    return u.select_atoms("name O and resname WAT and resid 1-10")


@pytest.fixture(scope='module')
def NSTEP():
    nstep = 5000
    return nstep


@pytest.fixture(scope='module')
def vacf(ag):
    # non-fft VACF
    v = VACF(ag, fft=False)
    v.run()
    return v


# Step trajectory of unit velocities i.e. v = 0 at t = 0,
# v = 1 at t = 1, v = 2 at t = 2, etc. for all components x, y, z
@pytest.fixture(scope='module')
def step_vtraj(NSTEP):
    v = np.arange(NSTEP)
    velocities = np.vstack([v, v, v]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5000
    velocities_reshape = velocities.reshape([NSTEP, 1, 3])
    u = mda.Universe.empty(1, n_frames=NSTEP, velocities=True)
    for i, ts in enumerate(u.trajectory):
        u.atoms.velocities = velocities_reshape[i]
    return u


# Expected VACF results for step_vtraj
# At time t, VACF is:
# sum_{x=0}^{N - 1 - t} x*(x + t) * n_dim / n_frames
# n_dim = 3 (typically) and n_frames = total_frames - t
def characteristic_poly(total_frames, n_dim):
    result = np.zeros(total_frames)
    for t in range(total_frames):
        sum = 0
        sum = np.dtype('float64').type(sum)
        for x in range((total_frames - t)):
            sum += x * (x + t)
        vacf = sum * n_dim / (total_frames - t)
        result[t] = vacf
    return result


@block_import('tidynamics')
def test_notidynamics(ag):
    with pytest.raises(ImportError, match="tidynamics was not found"):
        vacf = VACF(ag)
        vacf.run()


class TestVelocityAutocorr:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    def test_ag_accepted(self, ag):
        VACF(ag, fft=False)

    def test_no_velocities(self):
        u_no_vels = mda.Universe.empty(10, n_frames=5, velocities=False)
        errmsg = "VACF computation requires velocities"
        with pytest.raises(NoDataError, match=errmsg):
            v = VACF(u_no_vels.atoms, fft=False)
            v.run()

    def test_updating_ag_rejected(self, u):
        updating_ag = u.select_atoms("around 3.5 resid 1", updating=True)
        errmsg = "UpdatingAtomGroups are not valid"
        with pytest.raises(TypeError, match=errmsg):
            VACF(updating_ag, fft=False)

    @pytest.mark.parametrize('dimtype', ['foo', 'bar', 'yx', 'zyx'])
    def test_dimtype_error(self, ag, dimtype):
        errmsg = f"invalid dim_type: {dimtype}"
        with pytest.raises(ValueError, match=errmsg):
            VACF(ag, dim_type=dimtype)

    @pytest.mark.parametrize("tdim, tdim_factor", [
        ('xyz', 3), ('xy', 2), ('xz', 2), ('yz', 2), ('x', 1), ('y', 1),
        ('z', 1)
    ])
    def test_simple_step_vtraj_all_dims(self, step_vtraj, NSTEP, tdim,
                                        tdim_factor):
        # testing the "simple" windowed algorithm on unit velocity trajectory
        # VACF results should fit the polynomial defined in
        # characteristic_poly()
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        poly = characteristic_poly(NSTEP, tdim_factor)
        assert_almost_equal(v_simple.results.timeseries, poly, decimal=4)


@pytest.mark.skipif(import_not_available("tidynamics"),
                    reason="Test skipped because tidynamics not found")
class TestVACFFFT(object):

    @pytest.fixture(scope='class')
    def vacf_fft(self, ag):
        # fft VACF
        v = VACF(ag, fft=True)
        v.run()
        return v

    @pytest.mark.parametrize("tdim, tdim_factor", [
        ('xyz', 3), ('xy', 2), ('xz', 2), ('yz', 2), ('x', 1), ('y', 1),
        ('z', 1)
    ])
    def test_fft_step_vtraj_all_dims(self, step_vtraj, NSTEP,
                                     tdim, tdim_factor):
        # testing the fft algorithm on unit velocity trajectory
        # defined in step_vtraj()
        # VACF results should fit the characteristic polynomial defined in
        # characteristic_poly()

        # fft based tests require a slight decrease in expected prescision
        # primarily due to roundoff in fft(ifft()) calls.
        # relative accuracy expected to be around ~1e-12
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_simple.run()
        poly = characteristic_poly(NSTEP, tdim_factor)
        # this was relaxed from decimal=4 for numpy=1.13 test
        assert_almost_equal(v_simple.results.timeseries, poly, decimal=3)

    def test_fft_vs_simple_default(self, vacf, vacf_fft):
        # testing on the PRM_NCBOX, TRJ_NCBOX trajectory
        timeseries_simple = vacf.results.timeseries
        timeseries_fft = vacf_fft.results.timeseries
        assert_almost_equal(timeseries_simple, timeseries_fft, decimal=4)

    def test_fft_vs_simple_default_per_particle(self, vacf, vacf_fft):
        # check fft and simple give same result per particle
        per_particle_simple = vacf.results.vacf_by_particle
        per_particle_fft = vacf_fft.results.vacf_by_particle
        assert_almost_equal(per_particle_simple, per_particle_fft, decimal=4)
