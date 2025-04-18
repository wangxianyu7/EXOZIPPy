import unittest
import numpy as np
from exozippy.mmexofast import estimate_params


class KB160625():
    """
    Parameters for KMT-2016-BLG-0625 Shin et al. 2024.
    """
    def __init__(self):
        # Section 3.4
        self.tau_pl = 0.609
        self.u_pl = 0.613
        self.t_pl = 7662.95
        self.t_E = 11.5
        self.s_close = 0.739
        self.s_wide = 1.352
        self.alpha = [0.12, 3.26]  # radians

        # Data from Table 5:
        # s-
        self.close_params = {'t_0': 7655.951, 'u_0': 0.073, 't_E': 11.494,
                             's': 0.741, 'q': 2.357e-4, 'alpha': np.rad2deg(3.217),
                             'rho': 1.2256e-3}
        # s+
        self.wide_params = {'t_0': 7655.951, 'u_0': 0.075, 't_E':11.335,
                            's': 1.367, 'q': 0.727e-4, 'alpha': np.rad2deg(0.122), 'rho': 1.7656e-3}
        # 1L2S
        self.binary_source_params = {
            't_0_1': 7655.953, 'u_0_1': 0.078, 't_E': 10.946, 't_0_2': 7662.943, 'u_0_2': 3.751e-4, 'rho_2': 5.1309e-3, 'q_flux':  0.005
        }

        # dt and dmag estimated by-eye from figure in paper.
        self.dt = 0.25
        self.dmag = 0.3

        # Approximate values of t_0 and u_0
        self.t_0 = 7655.95
        self.u_0 = 0.075

        self.tol = 0.03  # 2% uncertainty based on variation in u0

        self.params = {
            't_0': self.t_0, 'u_0': self.u_0, 't_E': self.t_E, 't_pl': self.t_pl,
            'dt': self.dt, 'dmag': self.dmag
            }


class OB180383():
    """
    Parameters for OGLE-2018-BLG-0383 Wang et al. 2022.
    """
    def __init__(self):
        # Section 3.2

        # Table 2
        pass


class TestGetWideParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.ulens_params = estimate_params.get_wide_params(self.params)

    def test_pspl(self):
        """
        t_0, u_0, t_E
        """
        for key in ['t_0', 'u_0', 't_E']:
            assert self.ulens_params.ulens[key] == self.__getattribute__(key)

    def test_s(self):
        np.testing.assert_allclose(self.ulens_params.ulens['s'], self.s_wide, rtol=self.tol)

    def test_alpha(self):
        alpha = np.min(np.abs(self.ulens_params.ulens['alpha'] - np.rad2deg(self.alpha)))
        np.testing.assert_allclose(self.ulens_params.ulens['alpha'], alpha, rtol=self.tol)

    def test_q_rho(self):
        # Gould & Gaucherel approximation
        # Ap = 2(q / ρ^2)
        Ap_true = 2. * self.wide_params['q'] / self.wide_params['rho']**2
        Ap_est = 2. * self.ulens_params.ulens['q'] / self.ulens_params.ulens['rho']**2
        #print(Ap_true, Ap_est)
        print('JCY: This is a good idea for a test, but this event is not in this regime...')
        np.testing.assert_allclose(Ap_est, Ap_true, rtol=self.tol)

    def test_mag_methods(self):
        expected_values = [self.t_pl - 5. * self.dt, 'VBBL', self.t_pl + 5. * self.dt]
        for actual, expected in zip(self.ulens_params.mag_method, expected_values):
            if isinstance(actual, str):
                assert actual == expected
            else:
                np.testing.assert_allclose(actual, expected, atol=0.001)


class TestGetCloseParams(unittest.TestCase, KB160625):

    def setUp(self):
        KB160625.__init__(self)
        self.ulens_params = estimate_params.get_close_params(self.params, q=self)


def test_model_pspl_at_pl():
    raise NotImplementedError()


class TestBinarySourceParams(unittest.TestCase):

    def test_set_source_flux_ratio(self):
        raise NotImplementedError()


def test_get_binary_source_params():
    raise NotImplementedError()
