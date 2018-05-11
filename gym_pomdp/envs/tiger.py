from enum import Enum

import gym
import numpy as np
from gym.spaces import Discrete

from gym_pomdp.envs.gui import TigerGui


class Obs(Enum):
    LEFT = 0
    RIGHT = 1
    NULL = 2


class State(Enum):
    LEFT = 0
    RIGHT = 1


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    LISTEN = 2


def state_to_str(state):
    if state == 0:
        return "left"
    elif state == 1:
        return "right"
    else:
        raise NotImplementedError()


def action_to_str(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "right"
    elif action == 2:
        return "listen"
    else:
        raise NotImplementedError()


class TigerEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, seed=0, correct_prob=.85):
        self.correct_prob = correct_prob
        self.action_space = Discrete(len(Action))
        self.state_space = Discrete(len(State))
        self.observation_space = Discrete(len(Obs))
        self._discount = .95
        self._reward_range = 10
        self._query = 0
        self.seed(seed)

    def reset(self):
        self.done = False
        self.t = 0
        self._query = 0
        self.state = self.state_space.sample()
        self.last_action = Action.LISTEN.value
        return Obs.NULL.value

    def seed(self, seed=1234):
        np.random.seed(seed)
        return [seed]

    def step(self, action):

        assert self.action_space.contains(action)
        assert self.done is False
        self.t += 1
        self._query += 1
        self.last_action = action

        rw = TigerEnv._compute_rw(self.state, action)
        if TigerEnv._is_terminal(self.state, action):
            self.done = True
            return self.state, rw, self.done, {'state': self.state}

        self._sample_state(action)
        ob = TigerEnv._sample_ob(action, self.state)
        self.done = False
        return ob, rw, self.done, {"state": self.state}

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            if not hasattr(self, "gui"):
                self.gui = TigerGui()
            msg = "A: " + action_to_str(self.last_action) + " S: " + state_to_str(self.state)
            self.gui.render(state=(self.last_action, self.state), msg=msg)
        elif mode == "ansi":
            print("Current step: {}, tiger is in state: {}, action took: {}".format(self.t, self.state,
                                                                                    self.last_action[0]))
        else:
            raise NotImplementedError()

    def close(self):
        self._render(close=True)

    def _set_state(self, state):
        self.state = state
        self.done = False

    def _generate_legal(self):
        return list(range(self.action_space.n))

    def _generate_preferred(self, history):
        return self._generate_legal()

    def _sample_state(self, action):
        if action == Action.RIGHT.value or action == Action.LEFT.value:
            self.state = self.state_space.sample()

    def _get_init_state(self):
        # fix initial belief to be exact
        return self.state_space.sample()

    @staticmethod
    def _compute_prob(action, next_state, ob, correct_prob=.85):
        p_ob = 0.0
        if action == Action.LISTEN.value and ob != Obs.NULL.value:
            if (next_state == State.LEFT.value and ob == Obs.LEFT.value) or (
                    next_state == State.RIGHT.value and ob == Obs.RIGHT.value):
                p_ob = correct_prob
            else:
                p_ob = 1 - correct_prob
        elif action != Action.LISTEN.value and ob == Obs.NULL.value:
            p_ob = 1.

        assert p_ob >= 0.0 and p_ob <= 1.0
        return p_ob

    @staticmethod
    def _sample_ob(action, next_state, correct_prob=.85):
        ob = Obs.NULL.value
        p = np.random.uniform()
        if action == Action.LISTEN.value:
            if next_state == State.LEFT.value:
                ob = Obs.RIGHT.value if p > correct_prob else Obs.LEFT.value
            else:
                ob = Obs.LEFT.value if p > correct_prob else Obs.RIGHT.value
        return ob

    @staticmethod
    def _local_move(state, last_action, last_ob):
        raise NotImplementedError()

    @staticmethod
    def _is_terminal(state, action):
        is_terminal = False
        if action != Action.LISTEN.value:
            is_terminal = (
                    (action == Action.LEFT.value and state == State.LEFT.value) or (
                    action == Action.RIGHT.value and state == State.RIGHT.value))
        return is_terminal

    @staticmethod
    def _compute_rw(state, action):
        if action == Action.LISTEN.value:
            reward = -1
        elif not TigerEnv._is_terminal(state, action):
            reward = 10
        else:
            reward = -20
        return reward


if __name__ == '__main__':
    env = TigerEnv(seed=100)
    rws = 0
    t = 0
    done = False
    env.reset()

    env.render()
    while not done:
        action = env.action_space.sample()
        ob1, r, done, info = env.step(action)
        env.render()

        rws += r
        t += 1
    env.close()
    print("Ep done with rw {} and t {}".format(rws, t))
