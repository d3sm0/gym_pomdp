from enum import Enum

import numpy as np
from gym import Env
from gym.spaces import Discrete

from gym_pomdp.envs.coord import Coord, Grid, Moves
from gym_pomdp.envs.gui import RockGui


class Obs(Enum):
    NULL = 0
    GOOD = 1
    BAD = -1


class Action(Enum):
    UP = 0
    RIGHT = 1  # east
    DOWN = 2
    LEFT = 3  # west
    SAMPLE = 4


def action_to_str(action):
    if action == 0:
        return "up"
    elif action == 1:
        return "right"
    elif action == 2:
        return "down"
    elif action == 3:
        return "left"
    elif action == 4:
        return "sample"
    elif action > 4:
        return "check"
    else:
        raise NotImplementedError()


config = {
    2: {"size": (2, 1),
        "init_pos": (0, 0),
        "rock_pos": [[1, 0]]},
    4: {"size": (4, 3),
        "init_pos": (2, 1),
        "rock_pos": [[1, 0], [3, 1], [2, 3]]},
    7: {"size": (7, 8),
        "init_pos": (0, 3),
        "rock_pos": [[2, 0], [0, 1], [3, 1], [6, 3], [2, 4], [3, 4], [5, 5], [1, 6]]},
    15: {"size": (15, 15),
         "init_pos": (0, 5),
         "rock_pos":
             [[0, 7, ], [0, 3, ], [1, 2, ], [1, 2, ], [2, 6, ], [3, 7, ], [3, 2, ], [4, 7, ], [5, 2, ],
              [6, 9, ],
              [9, 7, ], [9, 1, ], [11, 8, ], [13, 10, ], [14, 9, ],
              [12, 2, ]]},
}


# each rock is a vector of [pos, good/bad]
# the full state is x_size * y_size + n_rocks +
# the full state is [agent_x, agent_y,


# class RockGrid(Grid):
#     def __init__(self, board_size):
#         super().__init__(*board_size)
#         self.build_board(value=-1)


class Rock(object):
    def __init__(self, pos):
        self.status = int(np.sign(np.random.uniform(0, 1) - .5))
        self.pos = pos
        self.count = 0
        self.measured = 0
        self.lkw = 1.  # likely worthless
        self.lkv = 1.  # likely valuable
        self.prob_valuable = .5


class RockState(object):
    def __init__(self, pos):
        self.agent_pos = pos
        self.rocks = []
        self.target = -1  # Coord(-1,-1) # -1  # Coord(-1, -1)


# remove illegal termination
# add a stochastic version of rock sampling
class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=7, num_rocks=8, use_heuristic=False):

        assert board_size in list(config.keys()) and num_rocks in config[board_size]['size']

        self.num_rocks = num_rocks
        self._use_heuristic = use_heuristic

        self._rock_pos = [Coord(*rock) for rock in config[board_size]['rock_pos']]
        self._agent_pos = Coord(*config[board_size]['init_pos'])
        self.grid = Grid(board_size, board_size)

        for idx, rock in enumerate(self._rock_pos):
            self.grid.board[rock] = idx

        self.action_space = Discrete(len(Action) + self.num_rocks)
        self.observation_space = Discrete(len(Obs))
        self._discount = .95
        self._reward_range = 20
        self._penalization = -100
        self._query = 0

    def step(self, action):

        assert self.action_space.contains(action)
        assert self.done is False

        self.last_action = action
        self._query += 1

        reward = 0
        ob = Obs.NULL.value

        if action < Action.SAMPLE.value:
            if action == Action.RIGHT.value:
                if self.state.agent_pos.x + 1 < self.grid.x_size:
                    self.state.agent_pos += Moves.RIGHT.value
                else:
                    reward = 10
                    self.done = True
                    return ob, reward, self.done, {"state": self._encode_state(self.state)}
            elif action == Action.UP.value:
                if self.state.agent_pos.y + 1 < self.grid.y_size:
                    self.state.agent_pos += Moves.UP.value
                else:
                    reward = self._penalization
            elif action == Action.DOWN.value:
                if self.state.agent_pos.y - 1 >= 0:
                    self.state.agent_pos += Moves.DOWN.value
                else:
                    reward = self._penalization
            elif action == Action.LEFT.value:
                if self.state.agent_pos.x - 1 >= 0:
                    self.state.agent_pos += Moves.LEFT.value
                else:
                    reward = self._penalization
            else:
                raise NotImplementedError()

        if action == Action.SAMPLE.value:
            rock = self.grid[self.state.agent_pos]
            if rock >= 0 and not self.state.rocks[rock].status == 0:  # collected
                if self.state.rocks[rock].status == 1:
                    reward = 10
                else:
                    reward = -10
                self.state.rocks[rock].status = 0
            else:
                reward = self._penalization

        if action > Action.SAMPLE.value:
            rock = action - Action.SAMPLE.value - 1
            assert rock < self.num_rocks

            ob = self._sample_ob(self.state.agent_pos, self.state.rocks[rock])

            self.state.rocks[rock].measured += 1

            eff = self._efficiency(self.state.agent_pos, self.state.rocks[rock].pos)

            if ob == Obs.GOOD.value:
                self.state.rocks[rock].count += 1
                self.state.rocks[rock].lkv *= eff
                self.state.rocks[rock].lkw *= (1 - eff)
            else:
                self.state.rocks[rock].count -= 1
                self.state.rocks[rock].lkw *= eff
                self.state.rocks[rock].lkv *= (1 - eff)

            denom = (.5 * self.state.rocks[rock].lkv) + (.5 * self.state.rocks[rock].lkw)
            try:
                self.state.rocks[rock].prob_valuable = (.5 * self.state.rocks[rock].lkv) / denom
            except RuntimeWarning:
                print("wtf")

        # if self.state.target < 0 or self.state.agent_pos == self.state.rocks[self.state.target]:
        #     self.state.target = self._select_target(self.state, self.grid.x_size)

        # self.done = all([rock.status == 0 for rock in self.state.rocks])  # self._penalization == reward
        self.done = self._penalization == reward
        return ob, reward, self.done, {"state": self._encode_state(self.state)}

    def _decode_state(self, state, as_array=False):

        agent_pos = Coord(*state['agent_pos'])
        rock_state = RockState(agent_pos)
        for r in state['rocks']:
            rock = Rock(pos=0)
            rock.__dict__.update(r)
            rock_state.rocks.append(rock)

        if as_array:
            rocks = []
            for rock in rock_state.rocks:
                rocks.append(rock.status)

            return np.concatenate([[self.grid.get_index(agent_pos)], rocks])

        return rock_state

    def _encode_state(self, state):
        # use dictionary for state encodign

        return _encode_dict(state)
        # rocks can take 3 values: -1, 1, 0 if collected

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            if not hasattr(self, "gui"):
                start_pos = self.grid.get_index(self.state.agent_pos)
                obj_pos = [(self.grid.get_index(rock.pos), rock.status) for rock in self.state.rocks]
                self.gui = RockGui((self.grid.x_size, self.grid.y_size), start_pos=start_pos, obj=obj_pos)

            if self.last_action > Action.SAMPLE.value:
                rock = self.last_action - Action.SAMPLE.value - 1
                print("Rock S: {} P:{}".format(self.state.rocks[rock].status, self.state.rocks[rock].pos))
            # msg = "Action : " + action_to_str(self.last_action) + " Step: " + str(self.t) + " Rw: " + str(self.total_rw)
            agent_pos = self.grid.get_index(self.state.agent_pos)
            self.gui.render(agent_pos)

    def reset(self):
        self.done = False
        self._query = 0
        self.last_action = Action.SAMPLE.value
        self.state = self._get_init_state(should_encode=False)
        return Obs.NULL.value

    def _set_state(self, state):
        self.done = False
        self.state = self._decode_state(state)

    def close(self):
        self.render(close=True)

    def _compute_prob(self, action, next_state, ob):

        next_state = self._decode_state(next_state)

        if action <= Action.SAMPLE.value:
            return int(ob == Obs.NULL.value)

        eff = self._efficiency(next_state.agent_pos, next_state.rocks[action - Action.SAMPLE.value - 1].pos)

        if ob == Obs.GOOD.value and next_state.rocks[action - Action.SAMPLE.value - 1].status == 1:
            return eff
        elif ob == Obs.BAD.value and next_state.rocks[action - Action.SAMPLE.value - 1].status == -1:
            return eff
        else:
            return 1 - eff

    def _get_init_state(self, should_encode=True):

        rock_state = RockState(self._agent_pos)
        for idx in range(self.num_rocks):
            rock_state.rocks.append(Rock(self._rock_pos[idx]))
        return self._encode_state(rock_state) if should_encode else rock_state

    def _generate_legal(self):
        legal = [Action.RIGHT.value]  # can always go east
        if self.state.agent_pos.y + 1 < self.grid.y_size:
            legal.append(Action.UP.value)

        if self.state.agent_pos.y - 1 >= 0:
            legal.append(Action.DOWN.value)
        if self.state.agent_pos.x - 1 >= 0:
            legal.append(Action.LEFT.value)

        rock = self.grid[self.state.agent_pos]
        if rock >= 0 and self.state.rocks[rock].status != 0:
            legal.append(Action.SAMPLE.value)

        for rock in self.state.rocks:
            assert self.grid[rock.pos] != -1
            if rock.status != 0:
                legal.append(self.grid[rock.pos] + 1 + Action.SAMPLE.value)
        return legal

    def _generate_preferred(self, history):
        if not self._use_heuristic:
            return self._generate_legal()

        actions = []

        # sample rocks with high likelihood of being good
        rock = self.grid[self.state.agent_pos]
        if rock >= 0 and self.state.rocks[rock].status != 0 and history.size:
            total = 0
            # history
            for t in range(history.size):
                if history[t].action == rock + 1 + Action.SAMPLE.value:
                    if history[t].ob == Obs.GOOD.value:
                        total += 1
                    elif history[t].ob == Obs.BAD.value:
                        total -= 1
            if total > 0:
                actions.append(Action.SAMPLE.value)
                return actions

        # process the rocks

        all_bad = True
        direction = {
            "north": False,
            "south": False,
            "west": False,
            "east": False
        }
        for idx in range(self.num_rocks):
            rock = self.state.rocks[idx]
            if rock.status != 0:
                total = 0
                for t in range(history.size):
                    if history[t].action == idx + 1 + Action.SAMPLE.value:
                        if history[t].ob == Obs.GOOD.value:
                            total += 1
                        elif history[t].ob == Obs.BAD.value:
                            total -= 1
                if total >= 0:
                    all_bad = False

                    if rock.pos.y > self.state.agent_pos.y:
                        direction['north'] = True
                    elif rock.pos.y < self.state.agent_pos.y:
                        direction['south'] = True
                    elif rock.pos.x < self.state.agent_pos.x:
                        direction['west'] = True
                    elif rock.pos.x > self.state.agent_pos.x:
                        direction['east'] = True

        if all_bad:
            actions.append(Action.RIGHT.value)
            return actions

        # generate a random legal move
        # do not measure a collected rock
        # do no measure a rock too often
        # do not measure clearly bad rocks
        # don't move in a direction that puts you closer to bad rocks
        # never sample a rock

        if self.state.agent_pos.y + 1 < self.grid.y_size and direction['north']:
            actions.append(Action.UP.value)

        if direction['east']:
            actions.append(Action.RIGHT.value)

        if self.state.agent_pos.y - 1 >= 0 and direction['south']:
            actions.append(Action.DOWN.value)

        if self.state.agent_pos.x - 1 >= 0 and direction['west']:
            actions.append(Action.LEFT.value)

        for idx, rock in enumerate(self.state.rocks):
            if not rock.status == 0 and rock.measured < 5 and abs(rock.count) < 2 and 0 < rock.prob_valuable < 1:
                actions.append(idx + 1 + Action.SAMPLE.value)

        if len(actions) == 0:
            return self._generate_legal()

        return actions

    def __dict2np__(self, state):
        idx = self.grid.get_index(Coord(*state['agent_pos']))
        rocks = []
        for rock in state['rocks']:
            rocks.append(rock['status'])
        return np.concatenate([[idx], rocks])

    @staticmethod
    def _efficiency(agent_pos, rock_pos, hed=20):
        # TODO check me
        d = Grid.euclidean_distance(agent_pos, rock_pos)
        eff = (1 + pow(2, -d / hed)) * .5
        return eff

    @staticmethod
    def _select_target(rock_state, x_size):
        best_dist = x_size * 2
        best_rock = -1  # Coord(-1, -1)
        for idx, rock in enumerate(rock_state.rocks):
            if rock.status != 0 and rock.count >= 0:
                d = Grid.manhattan_distance(rock_state.agent_pos, rock.pos)
                if d < best_dist:
                    best_dist = d
                    best_rock = idx  # rock.pos
        return best_rock

    @staticmethod
    def _sample_ob(agent_pos, rock, hed=20):
        eff = RockEnv._efficiency(agent_pos, rock.pos, hed=hed)
        if np.random.binomial(1, eff):
            return Obs.GOOD.value if rock.status == 1 else Obs.BAD.value
        else:
            return Obs.BAD.value if rock.status == 1 else Obs.GOOD.value

    # @staticmethod
    # def _local_move(state, last_action, last_ob):
    #     rock = np.random.choice(state.rocks)
    #     rock.valuable = not rock.valuable  # TODO Not really sure what is going on here
    #
    #     if last_action > Action.SAMPLE.value:  # check rock
    #         rock = last_action - Action.SAMPLE.value - 1
    #         new_ob = RockEnv._sample_ob(state.agent_pos, state.rocks[rock])
    #         if new_ob != last_ob:
    #             return False
    #         if last_ob == Obs.GOOD.value and new_ob == Obs.BAD.value:
    #             state.rocks[rock].count += 2
    #         elif last_ob == Obs.BAD.value and new_ob == Obs.GOOD.value:
    #             state.rocks[rock].count -= 2
    #     return True


def _encode_dict(state):
    enc_state = {}
    for k, v in vars(state).items():
        if isinstance(v, list):
            l = []
            for idx, t in enumerate(v):
                l.append(vars(t))
            v = l
        enc_state[k] = v
    return enc_state


def int_to_one_hot(idx, size):
    h = np.zeros(size, dtype=np.int32)
    h[int(idx)] = 1
    return h


if __name__ == "__main__":
    from gym_pomdp.envs.history import History

    history = History()
    env = RockEnv(board_size=7, num_rocks=8)
    env.reset()
    env.render()
    r = 0
    discount = 1.
    for i in range(400):
        action = np.random.choice(env._generate_preferred(history))  # np.random.choice(env._generate_legal(), 1)[0]
        env._generate_preferred(history)
        ob, rw, done, info = env.step(action)
        history.append(action, ob)
        env.render()
        r += rw * discount
        discount *= env._discount
        if done:
            break
    print(r)
