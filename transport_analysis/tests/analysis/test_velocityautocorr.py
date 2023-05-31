import pytest
from numpy.testing import assert_allclose

from transport_analysis.analysis.velocityautocorr import (
    VelocityAutocorr as VACF
)
import MDAnalysis as mda
from transport_analysis.tests.utils import make_Universe

from transport_analysis.data.files import PRM_NCBOX, TRJ_NCBOX


@pytest.fixture(scope='module')
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


@pytest.fixture(scope='module')
def ag(u):
    return u.select_atoms("backbone and name CA and resid 1-10")


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
