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


class RockGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)
        self.build_board(value=-1)


class Rock(object):
    def __init__(self, pos):
        # self.valuable = np.random.binomial(1, p=.5)
        self.status = int(np.sign(np.random.uniform(0, 1) - .5))
        # self.collected = False
        self.pos = pos
        # self.count = 0
        # self.measured = 0
        # self.lkw = 1.  # likely worthless
        # self.lkv = 1.  # likely valuable
        # self.prob_valuable = .5


class RockState(object):
    def __init__(self, pos):
        self.agent_pos = pos
        self.rocks = []
        # self.n_rocks = 0
        # self.target = Coord(-1, -1)


class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=7, num_rocks=8):

        self.use_problem_id = False
        if board_size in config.keys() and num_rocks == config[board_size]['size'][1]:
            self.use_problem_id = board_size
            self.board_size = board_size
            self.num_rocks = config[board_size]['size'][1]
        else:
            self.board_size = board_size
            self.num_rocks = num_rocks
        self.grid = RockGrid(board_size=(board_size, board_size))
        self.action_space = Discrete(len(Action) + self.num_rocks)
        self.observation_space = Discrete(len(Obs))
        self._discount = .95
        self._reward_range = 20
        self._penalization = -100

    def step(self, action):

        self.last_action = action
        self.t += 1
        reward = 0
        ob = Obs.NULL.value
        assert self.action_space.contains(action)
        assert self.done is False
        if action < Action.SAMPLE.value:
            if action == Action.RIGHT.value:
                if self.state.agent_pos.x + 1 < self.grid.x_size:
                    self.state.agent_pos += Moves.RIGHT.value
                else:
                    reward = 10
                    self.done = True
                    return ob, reward,self.done,{"state": self._encode_state(self.state)}
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
            # self.state.rocks[rock].measured += 1

            # eff = self._efficiency(self.state.agent_pos, self.state.rocks[rock].pos)

            # if ob == Obs.GOOD.value:
            #     self.state.rocks[rock].count += 1
            #     self.state.rocks[rock].lkv *= eff
            #     self.state.rocks[rock].lkw *= (1 - eff)
            # else:
            #     self.state.rocks[rock].count -= 1
            #     self.state.rocks[rock].lkw *= eff
            #     self.state.rocks[rock].lkv *= (1 - eff)
            #
            # denom = (.5 * self.state.rocks[rock].lkv) + (.5 * self.state.rocks[rock].lkw)
            # self.state.rocks[rock].prob_valuable = (.5 * self.state.rocks[rock].lkv) / denom

        # p_ob = self._compute_prob(action, ob=ob, next_state=self.state)
        self.done = self._penalization == reward
        return ob, reward, self.done, {"state": self._encode_state(self.state)}

    def _decode_state(self, state):

        agent_pos = self.grid.get_coord(state[0])
        board = state[1:-self.num_rocks].reshape(self.grid.get_size)
        rock_status = state[-self.num_rocks:]
        rock_state = RockState(agent_pos)
        # assert board[board != -1].shape[0] == rock_status.shape[0]
        # true_pos = [self.grid.get_index(rock.pos) for rock in self.state.rocks]
        # true_status = [rock.status for rock in self.state.rocks]
        # _, pos_list = list(zip(*sorted(zip(board[board != -1], np.where(board.T.flatten() != -1)[0]))))
        pos_list = np.where(board.T.flatten() != -1)[0]

        # assert np.all(true_pos == pos_list)
        # assert np.all(true_status == rock_status)
        for pos, status in zip(pos_list, rock_status):
            rock = Rock(pos=self.grid.get_coord(pos))
            rock.status = status
            rock_state.rocks.append(rock)

        return rock_state, board

    def _encode_state(self, state):
        # rocks can take 3 values: -1, 1, 0 if collected
        grid = self.grid.board.flatten()
        agent_idx = np.array((self.grid.get_index(state.agent_pos),), dtype=np.int32)
        rocks = np.zeros(self.num_rocks, dtype=np.int32)

        for idx, rock in enumerate(state.rocks):
            rocks[idx] = rock.status

        # assert np.all(rocks == [rock.status for rock in self.state.rocks])
        # assert len(grid[grid != -1]) == len(rocks)
        return np.concatenate([agent_idx, grid, rocks])

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == "human":
            if not hasattr(self, "gui"):
                start_pos = self.grid.get_index(self.state.agent_pos)
                obj_pos = [self.grid.get_index(rock.pos) for rock in self.state.rocks]
                self.gui = RockGui((self.grid.x_size, self.grid.y_size), start_pos=start_pos, obj_pos=obj_pos)

            msg = "Action : " + action_to_str(self.last_action) + " Step: " + str(self.t) + " Rw: " + str(self.total_rw)
            agent_pos = self.grid.get_index(self.state.agent_pos)
            self.gui.render(agent_pos, msg=msg)

    def reset(self):
        self.done = False
        self.t = 0
        self.total_rw = 0
        self.last_action = Action.SAMPLE.value
        self.state = self._get_init_state(should_encode=False)
        return Obs.NULL.value

    def _set_state(self, state):
        # self.reset()
        self.done = False
        self.state, board = self._decode_state(state)
        self.grid.board = board.reshape(self.grid.get_size)
        # self.grid.build_board(-1)
        # for idx, rock in enumerate(state.rocks):
        #     self.grid[rock.pos] = idx
        # self.state = state

    def close(self):
        self.render(close=True)

    def _compute_prob(self, action, next_state, ob):

        next_state, _ = self._decode_state(next_state)

        if action <= Action.SAMPLE.value:
            return int(ob == Obs.NULL.value)

        if ob != Obs.GOOD.value and ob != Obs.BAD.value:
            return 0

        eff = self._efficiency(next_state.agent_pos, next_state.rocks[action - Action.SAMPLE.value - 1].pos)
        if next_state.rocks[action - Action.SAMPLE.value - 1].status == 1:
            return eff
        else:
            return 1 - eff

    def _get_init_state(self, should_encode=True):

        self.grid.build_board(value=-1)

        if self.use_problem_id:
            init_state = Coord(*config[self.grid.x_size]["init_pos"])
            rock_pos = config[self.grid.x_size]["rock_pos"]
        else:
            init_state = self.grid.sample()
            rock_pos = []
        rock_state = RockState(init_state)
        if self.use_problem_id:
            for idx in range(self.num_rocks):
                pos = Coord(*rock_pos[idx])
                rock_state.rocks.append(Rock(pos))
                self.grid[pos] = idx
        else:
            for idx in range(self.num_rocks):
                while True:
                    pos = self.grid.sample()
                    if self.grid.board[pos] == -1:
                        break
                rock_state.rocks.append(Rock(pos))
                self.grid[pos] = idx

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

    @staticmethod
    def _efficiency(agent_pos, rock_pos, hed=20):
        # TODO check me
        d = Grid.euclidean_distance(agent_pos, rock_pos)
        eff = (1 + pow(2, -d / hed)) * .5
        return eff

    # @staticmethod
    # def _select_target(rock_state, x_size):
    #     best_dist = x_size * 2
    #     best_rock = Coord(-1, -1)
    #     for rock in rock_state.rocks:
    #         if not rock.collected and rock.sampled >= 0:
    #             d = Grid.manhattan_distance(rock_state.agent_pos, rock.pos)
    #             if d < best_dist:
    #                 best_dist = d
    #                 best_rock = rock.pos
    #     return best_rock

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


def int_to_one_hot(idx, size):
    h = np.zeros(size, dtype=np.int32)
    h[int(idx)] = 1
    return h


if __name__ == "__main__":
    env = RockEnv(board_size=7, num_rocks=8)
    for idx in range(10000):
        ob = env.reset()
        done = False
        t = 0
        env.render()
        for i in range(400):
            action = np.random.choice(env._generate_legal(), 1)[0]
            ob, rw, done, info = env.step(action)
            env._set_state(info["state"])
            env._generate_legal()
            env.render()
            t += 1
            if done:
                break

        print("rw {}, t{}".format(rw, t))
