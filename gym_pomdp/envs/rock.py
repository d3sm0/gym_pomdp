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


# TODO For each env create a starter state method
# TODO Separate grid from env
# TODO move stuff in static method
# TODO create encoding for states
# TODO add generate legal for each env
# TODO add prior construction to each env
# TODO remove tile everywhere
# TODO add methods required by the MCTS: generate_legal, reward, discount as property

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


class RockGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)
        self.build_board(value=-1)


class Rock(object):
    def __init__(self, pos):
        self.valuable = np.random.binomial(1, p=.5)
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
        self.target = Coord(-1, -1)


class RockEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_rocks=7):
        self.num_rocks = num_rocks
        self.grid = RockGrid(board_size=(8, 8))
        self.action_space = Discrete(len(Moves) + num_rocks)
        self.observation_space = Discrete(3)
        self._discount = .95
        self._reward_range = 20

    def _step(self, action):

        assert self.action_space.contains(action)
        # assert not self.done

        reward = 0
        obs = Obs.NULL.value
        self.last_action = action

        if not action in self._generate_legal():
            reward -= 100
            self.done = True
        elif action < 4:
            if action == 1 and self.grid.is_inside(self.state.agent_pos + Moves.get_coord(action)):
                self.state.agent_pos += Moves.get_coord(action)
            elif action == 1:
                reward += 10
                self.done = True  # try to escape left, and we let it go
            else:
                self.state.agent_pos += Moves.get_coord(action)
        elif action == 4:
            rock = self.grid[self.state.agent_pos]
            assert rock >= 0
            if rock >= 0 and not self.state.rocks[rock].collected:
                self.state.rocks[rock].collectd = True
                if self.state.rocks[rock].valuable:
                    reward += 10
                else:
                    reward -= 10
            else:
                reward -= 100

        elif action > 4:
            rock = action - len(Moves)
            assert rock < self.num_rocks and rock >= 0
            obs = self._sample_ob(self.state.agent_pos, self.state.rocks[rock])

            eff = RockEnv._sensor_correctnes(self.state.agent_pos, self.state.rocks[rock].pos)
            self.state.rocks[rock].update_prob(ob=obs, eff=eff)

        if self.state.target == Coord(-1, -1) or self.state.agent_pos == self.state.target:
            self.state.target = self._select_target(self.state, self.grid.x_size)

        # self.done = True if reward == -100 or not self.grid.is_inside(self.state.agent_pos) else False
        self.total_rw += reward

        # self.done = not self.grid.is_inside(self.state.agent_pos) or all(
        #     rock.collected for rock in self.state.rocks)
        return obs, reward, self.done, {"state": self.state}

    def _render(self, mode='human', close=False):
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

    def _reset(self):
        self.done = False
        self.t = 0
        self.total_rw = 0
        self.last_action = 4
        self.state = self._get_init_state()
        return Obs.NULL.value

    def _set_state(self, state):
        self.reset()
        self.grid.build_board(-1)
        for idx, rock in enumerate(state.rocks):
            self.grid[rock.pos] = idx
        self.state = state

    def _close(self):
        self.render(close = True)
    def _get_init_state(self):

        self.grid.build_board(value=-1)

        init_state = Coord(0, self.grid.y_size // 2)
        rock_state = RockState(init_state)
        for idx in range(self.num_rocks):
            pos = self.grid.sample()
            rock_state.rocks.append(Rock(pos))
            self.grid[pos] = idx
        rock_state.target = RockEnv._select_target(rock_state, x_size=self.grid.x_size)
        return rock_state

    # @staticmethod
    def _generate_legal(self):
        rock_state = self.state

        legal = [1]  # can always go east
        if rock_state.agent_pos.y + 1 < self.grid.y_size:
            legal.append(0)

        if rock_state.agent_pos.y - 1 >= 0:
            legal.append(2)
        if rock_state.agent_pos.x - 1 >= 0:
            legal.append(3)

        rock = self.grid[rock_state.agent_pos]
        if rock >= 0 and not rock_state.rocks[rock].collected:
            legal.append(4)

        for rock in rock_state.rocks:
            if not rock.collected:
                legal.append(self.grid[rock.pos] + len(Moves))
        # assert self.action_space.contains(max(legal))
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

    @staticmethod
    def _local_move(state, last_action, last_ob):
        rock = np.random.choice(state.rocks)
        rock.valuable = not rock.valuable  # TODO Not really sure what is going on here

        if last_action > 4:  # check rock
            rock = last_action - 4 - 1
            new_ob = RockEnv._sample_ob(state.agent_pos, state.rocks[rock])
            if new_ob != last_ob:
                return False
            if last_ob == Obs.GOOD.value and new_ob == Obs.BAD.value:
                state.rocks[rock].count += 2
            elif last_ob == Obs.BAD.value and new_ob == Obs.GOOD.value:
                state.rocks[rock].count -= 2
        return True


if __name__ == "__main__":
    env = RockEnv()
    ob = env.reset()
    done = False
    t = 0
    env.render()
    for i in range(200):
        action = env.action_space.sample()
        ob, rw, done, info = env.step(action)
        # env.render()
        t += 1
        if done:
            break

    print("rw {}, t{}".format(rw, t))
