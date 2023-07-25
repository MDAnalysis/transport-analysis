import pytest

from transport_analysis.viscosity import (
    ViscosityHelfand as VH,
)
import MDAnalysis as mda

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
