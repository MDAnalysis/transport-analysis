import pytest
from numpy.testing import assert_allclose

from transport_analysis.analysis.velocityautocorr import (
    VelocityAutocorr as VACF
)
import MDAnalysis as mda
import numpy as np

from MDAnalysisTests.datafiles import PRM_NCBOX, TRJ_NCBOX
from MDAnalysisTests.util import block_import, import_not_available


@pytest.fixture(scope='module')
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


@pytest.fixture(scope='module')
def ag(u):
    return u.select_atoms("backbone and name CA and resid 1-10")


@pytest.fixture(scope='module')
def NSTEP():
    nstep = 5000
    return nstep


@pytest.fixture(scope='module')
def step_traj(NSTEP):
    v = np.arange(NSTEP)
    velocities = np.vstack([v, v, v]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5000
    velocities_reshape = velocities.reshape([NSTEP, 1, 3])
    u = mda.Universe.empty(1, n_frames=NSTEP, velocities=True)
    for i, ts in enumerate(u.trajectory):
        u.atoms.velocities = velocities_reshape
    return u


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
        errmsg = "atomgroup must be from a trajectory with velocities"
        with pytest.raises(AttributeError, match=errmsg):
            VACF(u_no_vels.atoms, fft=False)

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
