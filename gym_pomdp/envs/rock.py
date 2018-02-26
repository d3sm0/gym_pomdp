from gym import Env
from gym.spaces import Discrete
import numpy as np
from enum import Enum
from gym_pomdp.envs.coord import Coord, Grid, Moves
from gym_pomdp.envs.gui import RockGui


class Obs(Enum):
    NULL = 0
    GOOD = 1
    BAD = -1


def action_to_str(action):
    if action == 0:
        return "up"
    elif action == 1:
        return "left"
    elif action == 2:
        return "right"
    elif action == 3:
        return "down"
    elif action == 4:
        return "sample"
    elif action > 4:
        return "check"
    else:
        raise NotImplementedError()


class RockGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)

    def set_values(self, value):
        for idx in range(self.n_tiles):
            self.board.append(value)


class Rock(object):
    def __init__(self, coord):
        self.valuable = np.random.binomial(1, p=.5)
        self.collected = False
        self.pos = coord
        self.sampled = 0
        self.measured = 0
        self.lkw = 1.  # likely worthless
        self.lkv = 1.  # likely valuable
        self.prob_valuable = .5

    def update_prob(self, ob, eff):

        self.measured += 1
        if ob == Obs.GOOD.value:
            self.sampled += 1
            self.lkv *= eff
            self.lkw *= 1 - eff
        else:
            self.sampled -= 1
            self.lkw *= eff
            self.lkv *= 1 - eff

        denom = .5 * self.lkv + .5 * self.lkw
        self.prob_valuable = (.5 * self.prob_valuable) / denom


class RockState(object):
    def __init__(self, coord):
        self.agent_pos = coord
        self.rocks = []
        self.n_rocks = 0
        self.hed = 20
        self.target = 0


class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=(8, 8), num_rocks=7):
        self.grid = RockGrid(board_size)
        self.num_rocks = num_rocks
        self.action_space = Discrete(len(Moves) + num_rocks)

    def _step(self, action):

        reward = 0
        obs = Obs.NULL.value
        self.last_action = action

        assert self.action_space.contains(action)

        if not action in self.admissible_actions():
            reward -= 100
        elif action < 4:
            if action == 1 and self.grid.is_inside(self.rock_state.agent_pos + Moves.get_coord(action)):
                self.rock_state.agent_pos += Moves.get_coord(action)
            elif action == 1:
                reward += 10
                self.done = True  # try to escape left, and we let it go
            else:
                self.rock_state.agent_pos += Moves.get_coord(action)
        elif action == 4:
            agent_idx = self.grid.get_index(self.rock_state.agent_pos)
            rock = self.grid.board[agent_idx]
            assert rock >= 0
            self.rock_state.rocks[rock].collectd = True
            if self.rock_state.rocks[rock].valuable:
                reward += 10
            else:
                reward -= 10
            # rock = self.grid.get_index(self.rock_state.agent_pos)
            # if rock >= 0 and self.rock_state.rocks[rock].collected:
        elif action > 4:
            rock = action - len(Moves)
            assert rock < self.num_rocks and rock >= 0
            ob = self.get_obs(self.rock_state.agent_pos, self.rock_state.rocks[rock])

            eff = RockEnv.sensor_correctnes(self.rock_state.agent_pos, self.rock_state.rocks[rock].pos)
            self.rock_state.rocks[rock].update_prob(ob=ob, eff=eff)
            # if rock >= 0 and not self.rock_state.rocks[rock].collected:
            # else:
            #     reward -= 100

        if self.rock_state.target == -1 or self.rock_state.agent_pos == self.grid.get_coord(self.rock_state.target):
            self.rock_state.target = self.select_target(self.rock_state, x_size=self.grid.x_size)

        # self.done = True if reward == -100 or not self.grid.is_inside(self.rock_state.agent_pos) else False
        self.total_rw += reward
        self.done = not self.grid.is_inside(self.rock_state.agent_pos) or all(
            rock.collected for rock in self.rock_state.rocks)
        return obs, reward, self.done, {"state": self.rock_state}

    def _render(self, mode='human', close=False):
        if mode == "human":
            if not hasattr(self, "gui"):
                agent_idx = self.grid.get_index(self.rock_state.agent_pos)
                obj_idx = [self.grid.get_index(rock.pos) for rock in self.rock_state.rocks]
                self.gui = RockGui((self.grid.x_size, self.grid.y_size), start_pos=agent_idx, obj_pos=obj_idx)
            msg = "Action : " + action_to_str(self.last_action) + " Step: " + str(self.t) + " Rw: " + str(self.total_rw)
            idx = self.grid.get_index(self.rock_state.agent_pos)
            self.gui.render(idx, msg=msg)

    def _reset(self):
        self.done = False
        self.t = 0
        self.total_rw = 0
        self.last_action = 4
        self._init_state()
        return Obs.NULL.value

    def _init_state(self):
        self.hed = 20
        self.rock_state = RockState(Coord(0, self.grid.y_size // 2))
        self.grid.set_values(value=-1)
        for idx in range(self.num_rocks):
            self.rock_state.rocks.append(Rock(self.grid.sample()))
            rock_idx = self.grid.get_index(self.rock_state.rocks[idx].pos)
            self.grid.board[rock_idx] = idx
        self.rock_state.target = self.select_target(self.rock_state, x_size=self.grid.x_size)

    def admissible_actions(self):

        legal = [1]  # can always go east
        if self.rock_state.agent_pos.y + 1 < self.grid.y_size:
            legal.append(0)

        if self.rock_state.agent_pos.y - 1 >= 0:
            legal.append(2)
        if self.rock_state.agent_pos.x - 1 >= 0:
            legal.append(3)

        agent_idx = self.grid.get_index(self.rock_state.agent_pos)
        rock = self.grid.board[agent_idx]
        if rock >= 0 and not self.rock_state.rocks[rock].collected:
            legal.append(4)

        for rock in self.rock_state.rocks:
            if not rock.collected:
                idx = self.grid.get_index(rock.pos)
                legal.append(self.grid.board[idx] + len(Moves))
        assert self.action_space.contains(max(legal))
        return legal

    @staticmethod
    def compute_rw(state, action):
        # TODO implement as static
        pass

    @staticmethod
    def sensor_correctnes(agent_pos, rock_pos, hed=20):
        d = Grid.euclidean_distance(agent_pos, rock_pos)
        eff = (1 + pow(2, -d / hed)) * .5
        return eff

    @staticmethod
    def select_target(rock_state, x_size):
        best_dist = x_size * 2
        best_rock = -1
        for idx, rock in enumerate(rock_state.rocks):
            if not rock.collected and rock.sampled >= 0:
                d = Grid.manhattan_distance(rock_state.agent_pos, rock.pos)
                if d < best_dist:
                    best_dist = d
                    best_rock = idx
        return best_rock

    @staticmethod
    def get_obs(agent_pos, rock, hed=20):
        eff = RockEnv.sensor_correctnes(agent_pos, rock.pos, hed=hed)
        if np.random.random() > eff:
            return Obs.GOOD.value if rock.valuable else Obs.BAD.value
        else:
            return Obs.BAD.value if rock.valuable else Obs.GOOD.value


if __name__ == "__main__":
    env = RockEnv()
    ob = env.reset()
    done = False
    t = 0
    env.render()
    for i in range(100):
        action = env.action_space.sample()
        ob, rw, done, info = env.step(action)
        env.render()
        t += 1
        if done:
            break

    print("rw {}, t{}".format(rw, t))
