from enum import Enum

import numpy as np
from gym.core import Env
from gym.spaces import Discrete


# restrict depth of the policy

class Obs(Enum):
    OFF = 0
    ON = 1
    NULL = 2


def print_action(action, n_machines):
    if action < n_machines * 2:
        machine, reboot = divmod(action, 2)
        return "M: {} A: {}".format(machine, reboot)
    else:
        return "Null"


class NetworkEnv(Env):
    metadata = {"render.modes": ["ansi"]}

    def __init__(self, n_machines=10, problem_type=3, depth=60):
        self._p = 0.1  # failure prob
        self._q = 0.33  # 33
        self._query = 0
        self._depth = 60
        self._n_machines = n_machines
        self.action_space = Discrete(n_machines * 2 + 1)
        self.observation_space = Discrete(len(Obs))
        self._discount = .95
        self._reward_range = n_machines * 2
        self.neighbours = self.make_3legs_neighbours(
            self._n_machines) if problem_type == 3 else self.make_ring_neighbours(n_machines)

    def seed(self, seed=None):
        np.random.seed(seed)

    def _compute_prob(self, action, next_state, ob):

        if action < self._n_machines * 2:
            machine, _ = divmod(action, 2)
            if next_state[machine] == ob:
                return self._p_ob
            else:
                return 1 - self._p_ob
        else:
            if ob == Obs.NULL.value:
                return 1.
        # this should never happen
        return 0

    @property
    def _p_ob(self):
        return .95

    def reset(self):
        self.done = False
        self.t = 0
        self._query = 0
        self.last_action = self._n_machines * 2
        # self.state = self._get_init_state()
        self.state = np.ones(self._n_machines, dtype=np.int8)
        self._server = 0
        return Obs.OFF.value

    def step(self, action):

        assert self.action_space.contains(action)
        assert self.done == False
        reward = 0
        self._query += 1
        ob = Obs.NULL.value
        n_failures = np.zeros(self._n_machines, dtype=np.int8)
        self.last_action = action
        self.t += 1

        for i in range(self._n_machines):
            for j in range(len(self.neighbours[i])):
                if self.state[self.neighbours[i][j]] == 0:
                    n_failures[i] = 1

        for idx in range(self._n_machines):
            if self.state[idx] == 1:
                if len(self.neighbours[idx]) > 2:
                    reward += 2
                else:
                    reward += 1

        for idx in range(self._n_machines):
            if self.state[idx]:
                if not n_failures[idx]:
                    self.state[idx] = 1 - np.random.binomial(1, p=self._p)
                else:
                    self.state[idx] = 1 - np.random.binomial(1, p=self._q)

        if action < self._n_machines * 2:
            machine, reboot = divmod(action, 2)
            if reboot:
                reward -= 2.5
                self.state[machine] = 1
                ob = np.random.binomial(1, p=self._p_ob)
            else:
                reward -= .1
                if np.random.binomial(1, p=self._p_ob):
                    ob = self.state[machine]
                else:
                    ob = 1 - self.state[machine]

        return ob, reward, self.done, {"state": self.state.copy()}

    def render(self, mode="ansi", close=False):
        if close:
            return
        print("N: {}, S: {}".format(self.state.sum(), self._server), print_action(self.last_action, self._n_machines),
              sep='\t')

    def _set_state(self, state):
        self.done = False
        self.state = state

    def _get_init_state(self):
        return np.ones(shape=self._n_machines, dtype=np.int8)

    def _generate_legal(self):
        return list(range(self.action_space.n))
        # actions = [self.action_space.n - 1]
        # for action in range(self.action_space.n - 1):
        #     machine, _ = divmod(action, 2)
        #     if self.state[machine] == 0:
        #         actions.append(action)
        # return actions

    def _generate_preferred(self, history):
        return self._generate_legal()

    def sample_action(self):
        return np.random.choice(self._generate_legal())

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

    env = NetworkEnv(n_machines=19, problem_type=3)
    eps = []
    seed = 0
    for idx in range(500):
        env.reset()
        env.seed(0)
        r = 0
        discount = 1
        for idx in range(60):
            action = 16 * 2  # np.random.choice(env._generate_legal())
            # action = env.action_space.sample()
            ob, rw, done, info = env.step(action)
            p_ob = env._compute_prob(action, info['state'], ob)
            # env.render(mode="ansi")
            # r+= rw
            r += discount * rw
            discount *= .95
        eps.append(r)
    print(sum(eps) / 500)
