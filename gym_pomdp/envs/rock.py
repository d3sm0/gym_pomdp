from gym import Env
from gym.spaces import Discrete
import numpy as np
from enum import Enum
from gym_pomdp.envs.coord import Coord, Grid, Moves, Tile
from gym_pomdp.envs.gui import RockGui


class Obs(Enum):
    NULL = 0
    GOOD = 1
    BAD = -1


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


class RockTile(Tile):
    def __init__(self, coord):
        super().__init__(coord)
        self.value = -1


class RockGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)
        self.build_board(Tile=RockTile)


class Rock(object):
    def __init__(self, pos):
        self.valuable = np.random.binomial(1, p=.5)
        self.observation_space = Discrete(len(Obs))
        self.collected = False
        self.pos = pos
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
    def __init__(self, pos):
        self.agent_pos = pos
        self.rocks = []
        self.n_rocks = 0
        self.hed = 20
        self.target = Coord(-1, -1)


class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=(8, 8), num_rocks=7):
        self.grid = RockGrid(board_size)
        self.num_rocks = num_rocks
        self.action_space = Discrete(len(Moves) + num_rocks)

    @staticmethod
    def compute_rw(state, action):
        # TODO implement as static
        pass

    def _step(self, action):

        assert self.action_space.contains(action)
        assert not self.done

        reward = 0
        obs = Obs.NULL.value
        self.last_action = action

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
            # agent_idx = self.grid.get_index(self.rock_state.agent_pos)
            # rock = self.grid.board[agent_idx].value
            rock = self.grid[self.rock_state.agent_pos].value
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
            ob = self._sample_ob(self.rock_state.agent_pos, self.rock_state.rocks[rock])

            eff = RockEnv._sensor_correctnes(self.rock_state.agent_pos, self.rock_state.rocks[rock].pos)
            self.rock_state.rocks[rock].update_prob(ob=ob, eff=eff)
            # if rock >= 0 and not self.rock_state.rocks[rock].collected:
            # else:
            #     reward -= 100

        if self.rock_state.target == Coord(-1, -1) or self.rock_state.agent_pos == self.rock_state.target:
            self.rock_state.target = self._select_target(self.rock_state, x_size=self.grid.x_size)

        # self.done = True if reward == -100 or not self.grid.is_inside(self.rock_state.agent_pos) else False
        self.total_rw += reward
        self.done = not self.grid.is_inside(self.rock_state.agent_pos) or all(
            rock.collected for rock in self.rock_state.rocks)
        return obs, reward, self.done, {"state": self.rock_state}

    def _render(self, mode='human', close=False):
        if mode == "human":
            if not hasattr(self, "gui"):
                start_pos = self.grid.get_index(self.rock_state.agent_pos)
                obj_pos = [self.grid.get_index(rock.pos) for rock in self.rock_state.rocks]
                self.gui = RockGui((self.grid.x_size, self.grid.y_size), start_pos=start_pos, obj_pos=obj_pos)

            msg = "Action : " + action_to_str(self.last_action) + " Step: " + str(self.t) + " Rw: " + str(self.total_rw)
            agent_pos = self.grid.get_index(self.rock_state.agent_pos)
            self.gui.render(agent_pos, msg=msg)

    def _reset(self):
        self.done = False
        self.t = 0
        self.total_rw = 0
        self.last_action = 4
        self._init_state()
        return Obs.NULL.value

    def _init_state(self):
        self.hed = 20
        init_state = Coord(0, self.grid.y_size // 2)
        self.rock_state = RockState(init_state)
        # self.grid.build_board(Tile=RockTile)
        for idx in range(self.num_rocks):
            # self.rock_state.rocks.append(Rock(self.grid.sample()))
            # rock_idx = self.grid.get_index(self.rock_state.rocks[idx].pos)
            # self.grid.board[rock_idx].value  = idx
            pos = self.grid.sample()
            self.rock_state.rocks.append(Rock(pos))
            self.grid[pos].value = idx
        self.rock_state.target = self._select_target(self.rock_state, x_size=self.grid.x_size)

    def admissible_actions(self):

        legal = [1]  # can always go east
        if self.rock_state.agent_pos.y + 1 < self.grid.y_size:
            legal.append(0)

        if self.rock_state.agent_pos.y - 1 >= 0:
            legal.append(2)
        if self.rock_state.agent_pos.x - 1 >= 0:
            legal.append(3)

        rock = self.grid[self.rock_state.agent_pos].value
        if rock >= 0 and not self.rock_state.rocks[rock].collected:
            legal.append(4)

        for rock in self.rock_state.rocks:
            if not rock.collected:
                legal.append(self.grid[rock.pos].value + len(Moves))
        assert self.action_space.contains(max(legal))
        return legal

    @staticmethod
    def _sensor_correctnes(agent_pos, rock_pos, hed=20):
        d = Grid.euclidean_distance(agent_pos, rock_pos)
        eff = (1 + pow(2, -d / hed)) * .5
        return eff

    @staticmethod
    def _select_target(rock_state, x_size):
        best_dist = x_size * 2
        best_rock = Coord(-1, -1)
        for rock in rock_state.rocks:
            if not rock.collected and rock.sampled >= 0:
                d = Grid.manhattan_distance(rock_state.agent_pos, rock.pos)
                if d < best_dist:
                    best_dist = d
                    best_rock = rock.pos
        return best_rock

    @staticmethod
    def _sample_ob(agent_pos, rock, hed=20):
        eff = RockEnv._sensor_correctnes(agent_pos, rock.pos, hed=hed)
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
