import pytest
from numpy.testing import assert_allclose

from transport_analysis.viscosity import (
    ViscosityHelfand as VH,
)
import MDAnalysis as mda
from MDAnalysis.units import constants
from MDAnalysis.transformations import set_dimensions
import numpy as np

from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import PRM_NCBOX, TRJ_NCBOX, PSF, DCD

from transport_analysis.data.files import ec_traj_trr, ec_top


@pytest.fixture(scope="module")
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


@pytest.fixture()
def u_ec():
    return mda.Universe(ec_top, ec_traj_trr)


@pytest.fixture(scope="module")
def ag(u):
    return u.select_atoms("name O and resname WAT and resid 1-10")


@pytest.fixture(scope="module")
def u_no_vels():
    return mda.Universe(PSF, DCD)


@pytest.fixture(scope="module")
def ag_no_vels(u_no_vels):
    return u_no_vels.select_atoms("backbone and name CA and resid 1-10")


@pytest.fixture(scope="module")
def NSTEP():
    nstep = 5001
    return nstep


@pytest.fixture(scope="module")
def visc_helfand(ag):
    vh_t = VH(ag)
    vh_t.run()
    return vh_t


# Full step trajectory of unit velocities i.e. v = 0 at t = 0,
# v = 1 at t = 1, v = 2 at t = 2, etc. for all components x, y, z
# with mass, positions, and volume
@pytest.fixture(scope="module")
def step_vtraj_full(NSTEP):
    # Set up unit velocities
    v = np.arange(NSTEP)
    velocities = np.vstack([v, v, v]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5001
    velocities_reshape = velocities.reshape([NSTEP, 1, 3])

    # Positions derived from unit velocity setup
    x = np.arange(NSTEP).astype(np.float64)
    # Since initial position and velocity are 0 and acceleration is 1,
    # position = 1/2t^2
    x *= x / 2
    positions = np.vstack([x, x, x]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5001
    positions_reshape = positions.reshape([NSTEP, 1, 3])
    u = mda.Universe.empty(1, n_frames=NSTEP, velocities=True)

    # volume of 8.0
    dim = [2, 2, 2, 90, 90, 90]
    for i, ts in enumerate(u.trajectory):
        u.atoms.velocities = velocities_reshape[i]
        u.atoms.positions = positions_reshape[i]
        set_dimensions(dim)(u.trajectory.ts)

    # mass of 16.0
    u.add_TopologyAttr("masses", [16.0])
    return u


def characteristic_poly_helfand(
    test_universe,
    stop,
    n_dim,
    temp_avg=300.0,
    mass=16.0,
    vol_avg=8.0,
    start=0,
    step=1,
):
    # update when mda 2.6.0 releases with typo fix (MDAnalysis Issue #4213)
    try:
        boltzmann = constants["Boltzmann_constant"]
    except KeyError:
        boltzmann = constants["Boltzman_constant"]

    d = stop - start
    frames_used = d // step + 1 if d % step != 0 else d / step
    frames_used = int(frames_used)

    result = np.zeros((frames_used))
    keys = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
    }
    velocities = np.zeros((frames_used, 1, n_dim))
    positions = np.zeros((frames_used, 1, n_dim))

    for i, ts in enumerate(test_universe.trajectory[start:stop:step]):
        velocities[i] = ts.velocities[:, keys[n_dim]]
        positions[i] = ts.positions[:, keys[n_dim]]

    for lag in range(1, frames_used):
        diff = mass * (
            velocities[:-lag, :, :] * positions[:-lag, :, :]
            - velocities[lag:, :, :] * positions[lag:, :, :]
        )

        sq_diff = np.square(diff).mean(axis=-1)
        result[lag] = np.mean(sq_diff, axis=0)

    result = result / (2 * boltzmann * vol_avg * temp_avg)
    return result


class TestViscosityHelfand:
    def test_ag_accepted(self, ag):
        VH(ag)

    def test_no_velocities(self, ag_no_vels):
        errmsg = "Helfand viscosity computation requires"
        with pytest.raises(NoDataError, match=errmsg):
            v = VH(ag_no_vels)
            v.run()

    def test_updating_ag_rejected(self, u):
        updating_ag = u.select_atoms("around 3.5 resid 1", updating=True)
        errmsg = "UpdatingAtomGroups are not valid"
        with pytest.raises(TypeError, match=errmsg):
            VH(updating_ag)

    @pytest.mark.parametrize("dimtype", ["foo", "bar", "yx", "zyx"])
    def test_dimtype_error(self, ag, dimtype):
        errmsg = f"invalid dim_type: {dimtype}"
        with pytest.raises(ValueError, match=errmsg):
            VH(ag, dim_type=dimtype)

    def test_ec_universe(self, u_ec):
        vh = VH(u_ec.atoms, linear_fit_window=(10, 40))
        vh.run()
        # vh.plot_viscosity_function()
        # the actual value is 2.56, not expected to be exact
        assert np.allclose(0.0256, vh.results.viscosity, atol=0.005)

        assert vh.results.timeseries is not None


@pytest.mark.parametrize(
    "tdim, tdim_factor",
    [
        ("xyz", 3),
        ("xy", 2),
        ("xz", 2),
        ("yz", 2),
        ("x", 1),
        ("y", 1),
        ("z", 1),
    ],
)
class TestAllDims:
    def test_step_vtraj_all_dims(
        self, step_vtraj_full, NSTEP, tdim, tdim_factor
    ):
        # Helfand viscosity results should agree with the unit velocity traj
        # defined in characteristic_poly_helfand()
        vis_h = VH(step_vtraj_full.atoms, dim_type=tdim)
        vis_h.run()
        poly = characteristic_poly_helfand(step_vtraj_full, NSTEP, tdim_factor)
        assert_allclose(vis_h.results.timeseries, poly)

    def test_start_stop_step_all_dims(
        self,
        step_vtraj_full,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # Helfand viscosity results should agree with the unit velocity traj
        # defined in characteristic_poly_helfand()
        # test start stop step is working correctly
        vis_h = VH(step_vtraj_full.atoms, dim_type=tdim)
        vis_h.run(start=tstart, stop=tstop, step=tstep)
        # polynomial must take offset start into account
        poly = characteristic_poly_helfand(
            step_vtraj_full, tstop, tdim_factor, start=tstart, step=tstep
        )
        assert_allclose(vis_h.results.timeseries, poly)
