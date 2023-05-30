import pytest
from numpy.testing import assert_allclose

from transport_analysis.analysis.velocityautocorr import VelocityAutocorr
from transport_analysis.tests.utils import make_Universe


class TestVelocityAutocorr:

    # fixtures are helpful functions that set up a test
    # See more at https://docs.pytest.org/en/stable/how-to/fixtures.html
    @pytest.fixture
    def universe(self):
        u = make_Universe(
            extras=("names", "resnames",),
            n_frames=3,
        )
        # create toy data to test assumptions
        for ts in u.trajectory:
            ts.positions[:ts.frame] *= -1
        return u

    @pytest.fixture
    def analysis(self, universe):
        return VelocityAutocorr(universe)

    @pytest.mark.parametrize(
        "select, n_atoms",  # argument names
        [  # argument values in a tuple, in order
            ("all", 125),
            ("index 0:9", 10),
            ("segindex 3:4", 50),
        ]
    )
    def test_atom_selection(self, universe, select, n_atoms):
        # `universe` here is the fixture defined above
        analysis = VelocityAutocorr(
            universe, select=select)
        assert analysis.atomgroup.n_atoms == n_atoms

    @pytest.mark.parametrize(
        "stop, expected_mean",
        [
            (1, 0),
            (2, 0.5),
            (3, 1)
        ]
    )
    def test_mean_negative_atoms(self, analysis, stop, expected_mean):
        # assert we haven't run yet and the result doesn't exist yet
        assert "mean_negative_atoms" not in analysis.results
        analysis.run(stop=stop)
        assert analysis.n_frames == stop

        # when comparing floating point values, it's best to use assert_allclose
        # to allow for floating point precision differences
        assert_allclose(
            analysis.results.mean_negative_atoms,  # computed data
            expected_mean,  # reference / desired data
            rtol=1e-07,  # relative tolerance
            atol=0,  # absolute tolerance
            err_msg="mean_negative_atoms is not correct",
        )
