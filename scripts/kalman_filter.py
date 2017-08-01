import numpy as np


class KalmanFilter(object):

    def __init__(self, delta_t,
                 initial_position,
                 initial_velocity,
                 acceleration,
                 calc_error_in_position,
                 calc_error_in_velocity,
                 obs_error_in_position,
                 obs_error_in_velocity):

        self.I = np.array([[1, 0], [0, 1]])
        self.A = np.array([[1, delta_t], [0, 1]])
        self.B = np.array([[.5 * (delta_t**2), delta_t]])

        self.previous_state = np.array(
            [[initial_position], [initial_velocity]])
        self.predicted_state = self._calculate_predicted_state(acceleration)

        self.measured_state = np.zeros((2, 1))

        self.previous_P = np.array([
            [calc_error_in_position**2,
                (calc_error_in_position * calc_error_in_velocity)],
            [(calc_error_in_position * calc_error_in_velocity),
             calc_error_in_velocity**2]
        ])

        self.R = np.array([
            [obs_error_in_position**2,
                (obs_error_in_position * obs_error_in_velocity)],
            [(obs_error_in_position * obs_error_in_velocity), obs_error_in_velocity**2]
        ])

        self.obs_Error = 0.

    def _adjust_measured_observation(self, observed_position, observed_velocity):
        obs_X = np.array([[observed_position], [observed_velocity]])
        self.observed_state = self.I * obs_X + self.obs_Error

    def _calculate_predicted_state(self, acceleration):
        self.predicted_state = (
            self.A * self.previous_state) + (self.B * acceleration)

    def _calculate_predicted_covariance(self):
        self.P = (self.A * self.previous_P * self.A.T)

    def _update_predicted_convariance(self):
        self.previous_P = np.array(
            (self.I - (self.kalman_gain * self.I)) * self.P)

    def _calculate_kalman_gain(self):
        self.kalman_gain = (self.P * self.I.T) / \
            ((self.I * self.P * self.I.T) + self.R)

    def _calculate_current_position(self):
        self.current_state = self.previous_state + \
            (self.kalman_gain * (self.observed_state - (self.I * self.previous_state)))
        self.previous_state = np.array(self.current_state)

    def get_current_position(self, observed_position, observed_velocity, acceleration):
        self._calculate_predicted_state(acceleration)
        self._calculate_predicted_covariance()
        self._calculate_kalman_gain()
        self._adjust_measured_observation(observed_position, observed_velocity)
        self._calculate_current_position()
        self._update_predicted_convariance()

        return self.previous_state[0][0]
