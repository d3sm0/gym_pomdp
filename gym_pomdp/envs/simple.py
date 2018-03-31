from gym.spaces import Discrete
from gym.core import Env


class TestEnv(Env):
    def __init__(self, action_size=5, observation_size=5, max_depth=20):
        self.action_space = Discrete(action_size)
        self.observation_space = Discrete(observation_size)
        self.max_depth = max_depth
        self._discount = .95
        self._reward_range = 1

    def reset(self):
        self.state = 0

    def _get_init_state(self):
        # self.state = 0
        return 0  # self.state

    def _set_state(self, state):
        self.state = state

    def step(self, action):
        if self.state < self.max_depth and action == 0:
            rw = 1.0
        else:
            rw = 0.0
        ob = self.observation_space.sample()
        self.state += 1
        return ob, rw, False, {"state": self.state, "p_ob": 1.}

    def optimal_value(self):
        discount = 1.
        total_rw = 0.
        for n in range(self.max_depth):
            total_rw += discount
            discount *= self._discount

        return total_rw

    def mean_value(self):
        discount = 1.
        total_rw = 0.
        for n in range(self.max_depth):
            total_rw += discount / self.action_space.n
            discount *= self._discount
        return total_rw
