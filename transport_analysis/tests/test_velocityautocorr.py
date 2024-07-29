import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_allclose,
    assert_approx_equal,
)

from transport_analysis.velocityautocorr import (
    VelocityAutocorr as VACF,
)
import MDAnalysis as mda
import numpy as np
import tidynamics
from scipy import integrate
import MDAnalysis.analysis.msd as msd
from scipy.stats import linregress

from MDAnalysis.exceptions import NoDataError
from MDAnalysisTests.datafiles import PRM_NCBOX, TRJ_NCBOX


@pytest.fixture(scope="module")
def u():
    return mda.Universe(PRM_NCBOX, TRJ_NCBOX)


@pytest.fixture(scope="module")
def ag(u):
    return u.select_atoms("name O and resname WAT and resid 1-10")


@pytest.fixture(scope="module")
def NSTEP():
    nstep = 5001
    return nstep


@pytest.fixture(scope="module")
def vacf(ag):
    # non-fft VACF
    v = VACF(ag, fft=False)
    v.run()
    return v


# Step trajectory of unit velocities i.e. v = 0 at t = 0,
# v = 1 at t = 1, v = 2 at t = 2, etc. for all components x, y, z
@pytest.fixture(scope="module")
def step_vtraj(NSTEP):
    v = np.arange(NSTEP)
    velocities = np.vstack([v, v, v]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5001
    velocities_reshape = velocities.reshape([NSTEP, 1, 3])
    u = mda.Universe.empty(1, n_frames=NSTEP, velocities=True)
    for i, ts in enumerate(u.trajectory):
        u.atoms.velocities = velocities_reshape[i]
    return u


# Position trajectory corresponding to unit velocity trajectory
@pytest.fixture(scope="module")
def step_vtraj_pos(NSTEP):
    x = np.arange(NSTEP).astype(np.float64)
    # Since initial position and velocity are 0 and acceleration is 1,
    # position = 1/2t^2
    x *= x / 2
    positions = np.vstack([x, x, x]).T
    # NSTEP frames x 1 atom x 3 dimensions, where NSTEP = 5001
    positions_reshape = positions.reshape([NSTEP, 1, 3])
    u_pos = mda.Universe.empty(1)
    u_pos.load_new(positions_reshape)
    return u_pos


# Expected VACF results for step_vtraj
# At time t, VACF is:
# sum_{x=0}^{N - 1 - t} x*(x + t) * n_dim / n_frames
# n_dim = 3 (typically) and n_frames = total_frames - t
def characteristic_poly(last, n_dim, first=0, step=1):
    diff = last - first
    frames_used = diff // step + 1 if diff % step != 0 else diff / step
    frames_used = int(frames_used)
    result = np.zeros(frames_used)
    for t in range(first, last, step):
        sum = 0
        sum = np.dtype("float64").type(sum)
        lagtime = t - first
        for x in range(first, (last - lagtime), step):
            sum += x * (x + lagtime)
        current_index = int(lagtime / step)
        vacf = sum * n_dim / (frames_used - current_index)
        result[current_index] = vacf
    return result


@pytest.mark.parametrize(
    "tdim, tdim_keys", [(1, [0]), (2, [0, 1]), (3, [0, 1, 2])]
)
def test_characteristic_poly(step_vtraj, NSTEP, tdim, tdim_keys):
    # test `characteristic_poly()` against `tidynamics.acf()``

    # expected result from tidynamics.acf()
    # n_particles should be 1 unless modifying the test
    n_particles = len(step_vtraj.atoms)
    # 2D array of frames x particles
    expected = np.zeros((NSTEP, n_particles))
    # 3D array of frames x particles x dimensions
    step_velocities = np.zeros((NSTEP, n_particles, tdim))

    for i, ts in enumerate(step_vtraj.trajectory):
        step_velocities[i] = step_vtraj.atoms.velocities[:, tdim_keys]

    for n in range(n_particles):
        expected[:, n] = tidynamics.acf(step_velocities[:, n, :])

    # average over n_particles
    expected = expected.mean(axis=1)

    # result from characteristic_poly()
    actual = characteristic_poly(NSTEP, tdim)

    # compare actual and expected
    assert_almost_equal(actual, expected, decimal=4)


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

    @pytest.mark.parametrize("dimtype", ["foo", "bar", "yx", "zyx"])
    def test_dimtype_error(self, ag, dimtype):
        errmsg = f"invalid dim_type: {dimtype}"
        with pytest.raises(ValueError, match=errmsg):
            VACF(ag, dim_type=dimtype)

    def test_plot_vacf(self, vacf):
        # Expected data to be plotted
        x_exp = vacf.times
        y_exp = vacf.results.timeseries

        # Actual data returned from plot
        (line,) = vacf.plot_vacf()
        x_act, y_act = line.get_xydata().T

        assert_allclose(x_act, x_exp)
        assert_allclose(y_act, y_exp)

    def test_plot_vacf_labels(self, vacf):
        # Expected labels
        x_exp = "Time (ps)"
        y_exp = "Velocity Autocorrelation Function (Å^2 / ps^2)"

        # Actual labels returned from plot
        (line,) = vacf.plot_vacf()
        x_act = line.axes.get_xlabel()
        y_act = line.axes.get_ylabel()

        assert x_act == x_exp
        assert y_act == y_exp

    def test_plot_vacf_custom_labels(self, vacf):
        # Expected labels
        x_exp = "Custom x-label"
        y_exp = "Custom VACF"

        # Actual labels returned from plot
        (line,) = vacf.plot_vacf(xlabel=x_exp, ylabel=y_exp)
        x_act = line.axes.get_xlabel()
        y_act = line.axes.get_ylabel()

        assert x_act == x_exp
        assert y_act == y_exp

    def test_plot_vacf_start_stop_step(self, vacf, start=1, stop=9, step=2):
        # Expected data to be plotted
        x_exp = vacf.times[start:stop:step]
        y_exp = vacf.results.timeseries[start:stop:step]

        # Actual data returned from plot
        (line,) = vacf.plot_vacf(start=start, stop=stop, step=step)
        x_act, y_act = line.get_xydata().T

        assert_allclose(x_act, x_exp)
        assert_allclose(y_act, y_exp)

    def test_plot_vacf_exception(self, step_vtraj):
        v = VACF(step_vtraj.atoms, fft=False)
        errmsg = "Analysis must be run"
        with pytest.raises(RuntimeError, match=errmsg):
            v.plot_vacf()

    def test_self_diffusivity_gk_exception(self, step_vtraj):
        v = VACF(step_vtraj.atoms, fft=False)
        errmsg = "Analysis must be run"
        with pytest.raises(RuntimeError, match=errmsg):
            v.self_diffusivity_gk()

    def test_self_diffusivity_gk_odd_exception(self, step_vtraj):
        v = VACF(step_vtraj.atoms, fft=False)
        errmsg = "Analysis must be run"
        with pytest.raises(RuntimeError, match=errmsg):
            v.self_diffusivity_gk_odd()

    def test_plot_running_integral(self, vacf):
        # Expected data to be plotted
        x_exp = vacf.times
        y_exp = np.zeros(vacf.n_frames)

        for i in range(1, vacf.n_frames):
            y_exp[i] = (
                integrate.trapezoid(
                    vacf.results.timeseries[: i + 1], vacf.times[: i + 1]
                )
                / vacf.dim_fac
            )

        # Actual data returned from plot
        (line,) = vacf.plot_running_integral()
        x_act, y_act = line.get_xydata().T

        assert_allclose(x_act, x_exp)
        assert_allclose(y_act, y_exp)

    def test_plot_running_integral_labels(self, vacf):
        # Expected labels
        x_exp = "Time (ps)"
        y_exp = "Running Integral of the VACF (Å^2 / ps)"

        # Actual labels returned from plot
        (line,) = vacf.plot_running_integral()
        x_act = line.axes.get_xlabel()
        y_act = line.axes.get_ylabel()

        assert x_act == x_exp
        assert y_act == y_exp

    def test_plot_running_integral_custom_labels(self, vacf):
        # Expected labels
        x_exp = "Custom x-label"
        y_exp = "Custom Running Integral"

        # Actual labels returned from plot
        (line,) = vacf.plot_running_integral(xlabel=x_exp, ylabel=y_exp)
        x_act = line.axes.get_xlabel()
        y_act = line.axes.get_ylabel()

        assert x_act == x_exp
        assert y_act == y_exp

    def test_plot_running_integral_start_stop_step(
        self, vacf, start=1, stop=9, step=2
    ):
        t_range = range(start, stop, step)
        # Expected data to be plotted
        x_exp = vacf.times[start:stop:step]
        y_exp = np.zeros(len(t_range))

        for i, j in enumerate(t_range):
            if i > 0:
                y_exp[i] = (
                    integrate.trapezoid(
                        vacf.results.timeseries[start : j + 1 : step],
                        vacf.times[start : j + 1 : step],
                    )
                    / vacf.dim_fac
                )

        # Actual data returned from plot
        (line,) = vacf.plot_running_integral(start=start, stop=stop, step=step)
        x_act, y_act = line.get_xydata().T

        assert_allclose(x_act, x_exp)
        assert_allclose(y_act, y_exp)

    def test_plot_running_integral_exception(self, step_vtraj):
        v = VACF(step_vtraj.atoms, fft=False)
        errmsg = "Analysis must be run"
        with pytest.raises(RuntimeError, match=errmsg):
            v.plot_running_integral()


class TestVACFFFT(object):
    @pytest.fixture(scope="class")
    def vacf_fft(self, ag):
        # fft VACF
        v = VACF(ag, fft=True)
        v.run()
        return v

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
    def test_simple_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing the "simple" windowed algorithm on unit velocity trajectory
        # VACF results should fit the polynomial defined in
        # characteristic_poly()
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        poly = characteristic_poly(NSTEP, tdim_factor)
        assert_almost_equal(v_simple.results.timeseries, poly, decimal=4)

    def test_simple_start_stop_step_all_dims(
        self,
        step_vtraj,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # testing the simple "windowed" algorithm on unit velocity trajectory
        # defined in step_vtraj()
        # test start stop step is working correctly
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run(start=tstart, stop=tstop, step=tstep)
        poly = characteristic_poly(
            tstop, tdim_factor, first=tstart, step=tstep
        )
        # polynomial must take offset start into account
        assert_almost_equal(v_simple.results.timeseries, poly, decimal=4)

    def test_self_diffusivity_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing self-diffusivity calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        # Integration results should match a separate integration method
        # Simpson is used for the check
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        sd_actual = v_simple.self_diffusivity_gk()
        sd_expected = (
            integrate.simpson(
                y=characteristic_poly(NSTEP, tdim_factor), x=range(NSTEP)
            )
            / tdim_factor
        )
        # 24307638750.0 (act) agrees with 24307638888.888885 (exp) to 8 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=8)

    def test_self_diffusivity_start_stop_step_all_dims(
        self,
        step_vtraj,
        NSTEP,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # testing self-diffusivity calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        # check that start, stop, step is working correctly
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        sd_actual = v_simple.self_diffusivity_gk(
            start=tstart, stop=tstop, step=tstep
        )
        sd_expected = (
            integrate.simpson(
                y=characteristic_poly(NSTEP, tdim_factor)[tstart:tstop:tstep],
                x=range(NSTEP)[tstart:tstop:tstep],
            )
            / tdim_factor
        )
        # 7705160166.66 (act) agrees with 7705162888.88 (exp) to 6 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=6)

    def test_self_diffusivity_odd_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing self-diffusivity (simpson) calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        sd_actual = v_simple.self_diffusivity_gk_odd()
        sd_expected = (
            integrate.trapezoid(
                characteristic_poly(NSTEP, tdim_factor), range(NSTEP)
            )
            / tdim_factor
        )
        # 24307638750.0 (exp) agrees with 24307638888.888885 (act) to 8 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=8)

    def test_self_diffusivity_odd_start_stop_step_all_dims(
        self,
        step_vtraj,
        NSTEP,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # testing self-diffusivity (simpson) calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        # check that start, stop, step is working correctly
        v_simple = VACF(step_vtraj.atoms, dim_type=tdim, fft=False)
        v_simple.run()
        sd_actual = v_simple.self_diffusivity_gk_odd(
            start=tstart, stop=tstop, step=tstep
        )
        sd_expected = (
            integrate.trapezoid(
                characteristic_poly(NSTEP, tdim_factor)[tstart:tstop:tstep],
                range(NSTEP)[tstart:tstop:tstep],
            )
            / tdim_factor
        )
        # 7705160166.66 (exp) agrees with 7705162888.88 (act) to 6 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=6)

    def test_fft_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing the fft algorithm on unit velocity trajectory
        # defined in step_vtraj()
        # VACF results should fit the characteristic polynomial defined in
        # characteristic_poly()

        # fft based tests require a slight decrease in expected prescision
        # primarily due to roundoff in fft(ifft()) calls.
        # relative accuracy expected to be around ~1e-12
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        poly = characteristic_poly(NSTEP, tdim_factor)
        # this was relaxed from decimal=4 for numpy=1.13 test
        assert_almost_equal(v_fft.results.timeseries, poly, decimal=3)

    def test_fft_start_stop_step_all_dims(
        self, step_vtraj, tdim, tdim_factor, tstart=10, tstop=1000, tstep=10
    ):
        # testing the fft algorithm on unit velocity trajectory
        # defined in step_vtraj()
        # test start stop step is working correctly
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run(start=tstart, stop=tstop, step=tstep)
        poly = characteristic_poly(
            tstop, tdim_factor, first=tstart, step=tstep
        )
        # polynomial must take offset start into account
        assert_almost_equal(v_fft.results.timeseries, poly, decimal=3)

    def test_fft_self_diffusivity_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing self-diffusivity calculated from the fft VACF of the
        # unit velocity trajectory
        # Integration results should match a separate integration method
        # Simpson is used for the check
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        sd_actual = v_fft.self_diffusivity_gk()
        sd_expected = (
            integrate.simpson(
                y=characteristic_poly(NSTEP, tdim_factor), x=range(NSTEP)
            )
            / tdim_factor
        )
        # 24307638750.0 (act) agrees with 24307638888.888885 (exp) to 8 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=8)

    def test_fft_self_diffusivity_start_stop_step_all_dims(
        self,
        step_vtraj,
        NSTEP,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # testing self-diffusivity calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        # check that start, stop, step is working correctly
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        sd_actual = v_fft.self_diffusivity_gk(
            start=tstart, stop=tstop, step=tstep
        )
        sd_expected = (
            integrate.simpson(
                y=characteristic_poly(NSTEP, tdim_factor)[tstart:tstop:tstep],
                x=range(NSTEP)[tstart:tstop:tstep],
            )
            / tdim_factor
        )
        # 7705160166.66 (act) agrees with 7705162888.88 (exp) to 6 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=6)

    def test_fft_self_diffusivity_odd_step_vtraj_all_dims(
        self, step_vtraj, NSTEP, tdim, tdim_factor
    ):
        # testing self-diffusivity (simpson) calculated from the fft VACF of the
        # unit velocity trajectory
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        sd_actual = v_fft.self_diffusivity_gk_odd()
        sd_expected = (
            integrate.trapezoid(
                y=characteristic_poly(NSTEP, tdim_factor), x=range(NSTEP)
            )
            / tdim_factor
        )
        # 24307638750.0 (exp) agrees with 24307638888.888885 (act) to 8 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=8)

    def test_fft_self_diffusivity_odd_start_stop_step_all_dims(
        self,
        step_vtraj,
        NSTEP,
        tdim,
        tdim_factor,
        tstart=10,
        tstop=1000,
        tstep=10,
    ):
        # testing self-diffusivity (simpson) calculated from the VACF of the
        # "simple" windowed algorithm on unit velocity trajectory
        # check that start, stop, step is working correctly
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        sd_actual = v_fft.self_diffusivity_gk_odd(
            start=tstart, stop=tstop, step=tstep
        )
        sd_expected = (
            integrate.trapezoid(
                y=characteristic_poly(NSTEP, tdim_factor)[tstart:tstop:tstep],
                x=range(NSTEP)[tstart:tstop:tstep],
            )
            / tdim_factor
        )
        # 7705160166.66 (exp) agrees with 7705162888.88 (act) to 6 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=6)

    def test_self_diffusivity_msd_all_dims(
        self, step_vtraj, step_vtraj_pos, tdim, tdim_factor
    ):
        # testing self-diffusivity calculated from the VACF (Green-Kubo)
        # against self-diffusivity calculated from the MSD (Einstein)

        # Green-Kubo self-diffusivity (actual)
        v_fft = VACF(step_vtraj.atoms, dim_type=tdim, fft=True)
        v_fft.run()
        sd_actual = v_fft.self_diffusivity_gk()

        # Einstein self-diffusivity (expected)
        MSD = msd.EinsteinMSD(step_vtraj_pos, select="all", msd_type=tdim)
        MSD.run()
        msd_res = MSD.results.timeseries
        lagtimes = np.arange(MSD.n_frames)
        start_time, end_time = 3000, 5000
        linear_model = linregress(
            lagtimes[start_time:end_time], msd_res[start_time:end_time]
        )
        sd_expected = linear_model.slope / (2 * tdim_factor)

        # 24307638750.0 (act) agrees with 24146066174.916477 (exp) to 2 sig figs
        assert_approx_equal(sd_actual, sd_expected, significant=2)
