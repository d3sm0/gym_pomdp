from gym.core import Env
from gym.spaces import Discrete
from enum import Enum
import numpy as np


class Obs(Enum):
    OFF = 0
    ON = 1
    NULL = 2


class NetworkEnv(Env):
    def __init__(self, n_machines=4, _type="ring"):
        self._p = .1  # failure prob
        self._q = 1 / 3
        self._n_machines = n_machines
        self.action_space = Discrete(n_machines)
        self.observation_space = Discrete(3)
        self._discount = .95
        self._reward_range = n_machines * 2
        self.neighbours = self.make_3legs_neighbours(
            self._n_machines) if _type == "ring" else self.make_ring_neighbours(n_machines)

    def _sample_prob(self):
        return .95

    def _reset(self):
        self.done = False
        self.t = 0
        self.state = self._get_init_state()
        return 0

    def _step(self, action):

        assert self.action_space.contains(action)
        reward = 0
        ob = 2
        n_failures = np.zeros(self._n_machines, dtype=np.int32)

        for i in range(self._n_machines):
            for j in range(len(self.neighbours[i])):
                if not self.state[self.neighbours[i][j]]:
                    n_failures[i] = 1

        for idx in range(self._n_machines):
            if not n_failures[idx]:
                self.state[idx] = 1 - np.random.binomial(1, p=self._p)
            else:
                self.state[idx] = 1 - np.random.binomial(1, p=self._q)

        for idx in range(self._n_machines):
            if self.state[idx]:
                if len(self.neighbours[idx]) > 2:  # server
                    reward += 2
                else:
                    reward += 1

        if action < self._n_machines * 2:
            machine = action // 2
            reboot = action % 2

            if reboot:
                reward -= 2.5
                self.state[machine] = 1
                ob = np.random.binomial(1, self._sample_prob())

            else:
                reward -= .1
                if np.random.binomial(1, self._sample_prob()):
                    ob = self.state[machine]
                else:
                    ob = 1 - self.state[machine]

        return ob, reward, self.done, {"state": self.state}

    def _render(self, mode='human', close=False):
        if close:
            return

    def _get_init_state(self):
        return np.ones(self._n_machines, dtype=np.int32)

    def _generate_legal(self):
        return list(range(self.action_space.n))

    @staticmethod
    def make_ring_neighbours(n_machines):
        neighbours = [[] for idx in range(n_machines)]
        for idx in range(n_machines):
            neighbours[idx].append((idx + 1) % n_machines)
            neighbours[idx].append((idx + n_machines - 1) % n_machines)
        return neighbours

    @staticmethod
    def make_3legs_neighbours(n_machines):
        assert n_machines >= 4 and n_machines % 3 == 1
        neighbours = [[] for idx in range(n_machines)]
        neighbours[0].append(1)
        neighbours[0].append(2)
        neighbours[0].append(3)

        for idx in range(1, n_machines):
            if idx < n_machines - 3:
                neighbours[idx].append(idx + 3)
            if idx <= 4:
                neighbours[idx].append(0)
            else:
                neighbours[idx].append(idx - 3)

        return neighbours


if __name__ == "__main__":

    env = NetworkEnv()
    env.reset()
    r = 0
    for idx in range(100):
        action = env.action_space.sample()
        ob, rw, done, _ = env.step(action)
        print(ob)
        r += rw

    # print(rw)
