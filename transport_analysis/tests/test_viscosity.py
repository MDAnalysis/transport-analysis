import pytest

from transport_analysis.viscosity import (
    ViscosityHelfand as VH,
)
import MDAnalysis as mda
from MDAnalysis.units import constants
from MDAnalysis.transformations import set_dimensions
import numpy as np

from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import PRM_NCBOX, TRJ_NCBOX, PSF, DCD


@pytest.fixture(scope="module")
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


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
    vh_t = VH(ag, fft=False)
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
    total_frames, n_dim, temp_avg=300.0, vol_avg=8.0
):
    result = np.zeros(total_frames)

    for lag in range(total_frames):
        sum = 0
        sum = np.dtype("float64").type(sum)

        for curr in range((total_frames - lag)):
            # mass * velocities * positions
            helf_diff = 16.0 * (curr + lag) * 1 / 2 * (
                (curr + lag) ** 2
            ) - 16.0 * curr * 1 / 2 * (curr**2)
            sum += helf_diff**2

        vis_helf = (
            sum
            * n_dim
            / (
                (total_frames - lag)
                * 2
                * constants["Boltzmann_constant"]
                * vol_avg
                * temp_avg
            )
        )

        result[lag] = vis_helf
    return result


class TestViscosityHelfand:
    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    def test_ag_accepted(self, ag):
        VH(ag, fft=False)

    def test_no_velocities(self, ag_no_vels):
        errmsg = "Helfand viscosity computation requires"
        with pytest.raises(NoDataError, match=errmsg):
            v = VH(ag_no_vels, fft=False)
            v.run()

    def test_updating_ag_rejected(self, u):
        updating_ag = u.select_atoms("around 3.5 resid 1", updating=True)
        errmsg = "UpdatingAtomGroups are not valid"
        with pytest.raises(TypeError, match=errmsg):
            VH(updating_ag, fft=False)

    @pytest.mark.parametrize("dimtype", ["foo", "bar", "yx", "zyx"])
    def test_dimtype_error(self, ag, dimtype):
        errmsg = f"invalid dim_type: {dimtype}"
        with pytest.raises(ValueError, match=errmsg):
            VH(ag, dim_type=dimtype)
