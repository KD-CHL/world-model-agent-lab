"""Observed residuals adjust the next planning cycle's velocity envelope."""
import math


class ResidualFeedback:
    def __init__(self, threshold_m=0.08, alpha=0.3, minimum_scale=0.25):
        if not (math.isfinite(threshold_m) and threshold_m > 0
                and 0 < alpha <= 1 and 0 < minimum_scale <= 1):
            raise ValueError('Invalid feedback parameters')
        self.threshold_m, self.alpha, self.minimum_scale = threshold_m, alpha, minimum_scale
        self.error = 0.0
        self.episode = None

    def update(self, predicted, observed, planner):
        if predicted.episode_id != observed.episode_id or predicted.step_id != observed.step_id:
            raise ValueError('Feedback prediction and observation are not aligned')
        if abs(predicted.sim_time_s - observed.sim_time_s) > 1e-6:
            raise ValueError('Feedback timestamps are not aligned')
        if self.episode != observed.episode_id:
            self.error = 0.0
            self.episode = observed.episode_id
        residual = math.hypot(predicted.x - observed.x, predicted.y - observed.y)
        self.error = self.alpha * residual + (1 - self.alpha) * self.error
        scale = max(self.minimum_scale, min(1.0, self.threshold_m / max(self.error, 1e-12)))
        planner.feedback_scale = scale
        return {'position_error_ema_m': self.error, 'velocity_scale': scale}
