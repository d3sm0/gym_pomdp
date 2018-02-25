import gym
from gym import spaces
from enum import Enum
import numpy as np
from .gui import TigerGui


class Obs(Enum):
    LEFT = 0
    RIGHT = 1
    NULL = 2

    @staticmethod
    def get_random_obs():
        return np.random.choice(Obs).value

    @staticmethod
    def get_name(idx):
        return list(Obs)[idx].name.lower()


class State(Enum):
    LEFT = 0
    RIGHT = 1

    @staticmethod
    def get_random_state():
        return np.random.choice(State).value

    @staticmethod
    def get_name(idx):
        return list(State)[idx].name.lower()


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    LISTEN = 2

    @staticmethod
    def get_random_action():
        return np.random.choice(Action).value

    @staticmethod
    def get_name(idx):
        return list(Action)[idx].name.lower()


class Tiger(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, seed=0, correct_prob=.85):
        self.correct_prob = correct_prob
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Discrete(len(Obs))
        self.state_space = spaces.Discrete(len(State))
        self.time = 0
        self.seed(seed)

    def _reset(self):
        self.done = False
        self.time = 0
        self.state = self.state_space.sample()
        self.last_action = Action.LISTEN.value
        return Obs.NULL.value

    def _seed(self, seed=1234):
        # TODO check if all seed are equal
        np.random.seed(seed)
        return [seed]

    def _step(self, action):

        assert self.action_space.contains(action)
        assert self.done is not True
        self.time += 1
        self.last_action = action

        rw = Tiger._compute_rw(self.state, action)
        if Tiger._is_terminal(self.state, action):
            return self.state, rw, True, dict(state=self.state, p_ob=1.)

        self.state = Tiger._sample_state(self.state, action)
        ob = Tiger._sample_ob(action, self.state)
        p_ob = Tiger._compute_prob(action, self.state, ob)
        return ob, rw, False, dict(state=self.state, p_ob=p_ob)

    def _render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            if not hasattr(self, "gui"):
                self.gui = TigerGui()
            msg = "A: " + Action.get_name(self.last_action) + " S: " + State.get_name(self.state)
            self.gui.render(state=(self.last_action, self.state), msg=msg)
        elif mode == "ansi":
            print("Current step: {}, tiger is in state: {}, action took: {}".format(self.time, self.state,
                                                                                    self.last_action[0]))
        else:
            raise NotImplementedError()

    def _close(self):
        self._render(close=True)

    def set_state(self, state):
        self.state = state

    @staticmethod
    def _sample_state(state, action):
        if action == Action.RIGHT.value or action == Action.LEFT.value:
            state = State.get_random_state()
        return state

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
        elif not Tiger._is_terminal(state, action):
            reward = 10
        else:
            reward = -100
        return reward


if __name__ == '__main__':
    env = Tiger(seed=100)
    rws = 0
    t = 0
    done = False
    env.reset()

    env.render()
    while not done:
        action = Action.get_random_action()
        ob1, r, done, info = env.step(action)
        env.render()

        rws += r
        t += 1
    env.close()
    print("Ep done with rw {} and t {}".format(rws, t))
