import numpy as np
from enum import Enum
from gym import Env
from gym.spaces import Discrete
from gym_pomdp.envs.coord import Grid, Coord, Moves, Tile


class Action(Enum):
    UP = 0
    RIGHT = 1  # east
    DOWN = 2
    LEFT = 3  # west


class PocState(object):
    def __init__(self, pos=(0, 0)):
        self.agent_pos = pos
        self.ghosts = []
        self.food_pos = []
        self.power_step = 0


class Ghost(object):
    def __init__(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def update(self, pos, direction):
        self.pos = pos
        self.direction = direction

    def reset(self):
        self.pos = Coord(*config["_ghost_home"])
        self.direction = -1


config = dict(
    _maze=np.array([
        [3, 3, 3, 3, 3, 3, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 3, 0, 3, 3, 3],
        [3, 0, 3, 3, 3, 0, 3],
        [3, 3, 0, 3, 0, 3, 3],
        [3, 3, 3, 3, 3, 3, 3]], dtype=np.int32),
    _num_ghosts=1,
    _ghost_range=3,
    _ghost_home=(3, 4),
    _poc_home=(3, 0),
    _passage_y=-1,
    _smell_range=1,
    _hear_range=2,
    _food_prob=.5,
    _chase_prob=.75,
    _defensive_slip=.25,
    _power_steps=15,
)


def check_flags(flags, bit):
    return (flags and (1 << bit)) != 0


def set_flags(flags, bit):
    return flags or 1 << bit


def can_move(ghost, d):
    return Grid.opposite(d) != ghost.direction


def is_power(idx):
    return config["_maze"][idx] == 3


def is_passable(idx):
    return config["_maze"][idx] != 0


def select_maze(_size):
    if _size == "micro":
        return config


class PocTile(Tile):
    def __init__(self, coord=(0, 0)):
        super().__init__(coord)


class PocGrid(Grid):
    def __init__(self, board_size):
        super().__init__(board_size[0], board_size[1])
        self.build_board(PocTile)


class PocEnv(Env):
    def __init__(self, config=config):

        self.grid = PocGrid(board_size=config["_maze"].shape)
        self._get_init_state(config)
        self.action_space = Discrete(4)
        self.observation_space = Discrete(1 << 10)  # 1024
        self._reward_range = 100
        self._discount = .95

    def _is_valid(self):

        assert self.grid.is_inside(self.state.agent_pos)
        assert is_passable(self.state.agent_pos)
        for ghost in self.state.ghosts:
            assert self.grid.is_inside(ghost.pos)
            assert is_passable(ghost.pos)

    def _set_state(self, state):
        self.done = False
        self.state = state

    def _generate_legal(self):
        actions = []
        for action in Action:
            if self.grid.is_inside(self.state.agent_pos + Moves.get_coord(action.value)):
                actions.append(action.value)
        return actions

    def _step(self, action):
        assert self.action_space.contains(action)
        assert self.done == False

        reward = -1
        next_pos = self.next_pos(self.state.agent_pos, action)
        if next_pos.is_valid():
            self.state.agent_pos = next_pos
        else:
            reward += -25

        if self.state.power_step > 0:
            self.state.power_step -= 1

        hit_ghost = -1
        for g, ghost in enumerate(self.state.ghosts):
            if ghost.pos == self.state.agent_pos:
                hit_ghost = g
            # move ghost
            self._move_ghost(g, ghost_range=config["_ghost_range"])

        if hit_ghost >= 0:
            if self.state.power_step > 0:
                reward += 25
                self.state.ghosts[hit_ghost].reset()

            else:
                reward += - 100
                self.done = True

        ob = self.make_ob(action)

        if self.state.food_pos[self.grid.get_index(self.state.agent_pos)]:
            if sum(self.state.food_pos) == 0:
                reward += 1000
                self.done = True
            if is_power(self.state.agent_pos):
                self.state.power_step = config["_power_steps"]
            reward += 10
        return ob, reward, self.done, {"state": self.state, "p_ob": self._compute_prob(action, self.state, ob)}

    def make_ob(self, action):
        ob = 0
        for d in range(self.action_space.n):
            if self._see_ghost(action) > 0:
                ob = set_flags(ob, d)
            next_pos = self.next_pos(self.state.agent_pos, direction=d)
            if next_pos.is_valid() and is_passable(next_pos):
                ob = set_flags(ob, d + self.action_space.n)
        if self._smell_food():
            ob = set_flags(ob, 8)
        if self._hear_ghost(self.state):
            ob = set_flags(ob, 9)
        return ob

    def _compute_prob(self, action, next_state, ob):
        return int(ob == self.make_ob(action))

    def _see_ghost(self, action):
        eye_pos = self.state.agent_pos + Moves.get_coord(action)
        while True:
            for g, ghost in enumerate(self.state.ghosts):
                if ghost.pos == eye_pos:
                    return g
            eye_pos += Moves.get_coord(action)
            if not self.grid.is_inside(eye_pos) or not is_passable(eye_pos):
                break
        return -1

    def _smell_food(self, smell_range=1):
        for x in range(-smell_range, smell_range + 1):
            for y in range(-smell_range, smell_range + 1):
                smell_pos = Coord(x, y)
                idx = self.grid.get_index(self.state.agent_pos + smell_pos)
                if self.grid.is_inside(self.state.agent_pos + smell_pos) and self.state.food_pos[idx]:
                    return True
        return False

    @staticmethod
    def _hear_ghost(poc_state, hear_range=2):
        for ghost in poc_state.ghosts:
            if Grid.manhattan_distance(ghost.pos, poc_state.agent_pos) <= hear_range:
                return True
        return False

    def _render(self, mode='human', close=False):
        pass

    def _reset(self):
        self.t = 0
        self.done = False
        self._get_init_state(config)
        return 0

    def _close(self):
        pass

    def _get_init_state(self, config=config):
        # create walls
        # for tile in self.grid:
        #     value = config["maze"][tile.key[0]]
        #     self.grid.set_value(value, coord=tile.key)

        self.state = PocState()
        self.state.agent_pos = Coord(*config["_poc_home"])
        for g in range(config["_num_ghosts"]):
            pos = Coord(config['_ghost_home'][0] + g % 2, config['_ghost_home'][1] + g // 2)
            # pos = Coord(*config["_ghost_home"])
            # pos.x += g % 2
            # pos.y += g // 2
            self.state.ghosts.append(Ghost(pos, direction=-1))

        self.state.food_pos = np.random.binomial(1, config["_food_prob"], size=self.grid.n_tiles + 1)
        self.state.power_step = 0
        return self.state

    def next_pos(self, pos, direction, passage_y=config["_passage_y"]):
        direction = Moves.get_coord(direction)
        if pos.x == 0 and pos.y == passage_y and direction == Moves.RIGHT:
            next_pos = Coord(self.grid.x_size - 1, pos.y)
        elif pos.x == self.grid.x_size - 1 and pos.y == passage_y and direction == Moves.LEFT:
            next_pos = Coord(0, pos.y)
        else:
            next_pos = pos + direction

        if self.grid.is_inside(next_pos) and is_passable(next_pos):
            return next_pos
        else:
            return Coord(-1, -1)

    def _move_ghost(self, g, ghost_range):
        if Grid.manhattan_distance(self.state.agent_pos, self.state.ghosts[g].pos) < ghost_range:
            if self.state.power_step > 0:
                self._move_defensive(g)
            else:
                self._move_aggressive(g)
        else:
            self._move_random(g)

    def _move_aggressive(self, g, chase_prob=.75):
        if not np.random.binomial(1, p=chase_prob):
            return self._move_random(g)

        best_dist = self.grid.x_size + self.grid.y_size
        best_pos = self.state.ghosts[g].pos
        best_dir = -1
        for d in range(self.action_space.n):
            dist = Grid.directional_distance(self.state.agent_pos, self.state.ghosts[g].pos, d)
            new_pos = self.next_pos(self.state.ghosts[g].pos, d)
            if dist <= best_dist and new_pos.is_valid() and can_move(self.state.ghosts[g], d):
                best_pos = new_pos
                best_dist = dist
                best_dir = d

        self.state.ghosts[g].update(best_pos, best_dir)

    def _move_defensive(self, g, defensive_prob=.5):
        if np.random.binomial(1, defensive_prob) and self.state.ghosts[g].direction >= 0:
            self.state.ghosts[g].direction = -1

        best_dist = self.grid.x_size + self.grid.y_size
        best_pos = self.state.ghosts[g].pos
        best_dir = -1
        for d in range(self.action_space.n):
            dist = Grid.directional_distance(self.state.agent_pos, self.state.ghosts[g].pos, d)
            new_pos = self.next_pos(self.state.ghosts[g].pos, d)
            if dist >= best_dist and new_pos.is_valid() and can_move(self.state.ghosts[g], d):
                best_pos = new_pos
                best_dist = dist
                best_dir = d

        self.state.ghosts[g].update(best_pos, best_dir)

    def _move_random(self, g):
        # there are no dead ends
        # never switch to opposite direction
        ghost_pos = self.state.ghosts[g].pos
        while True:
            d = self.action_space.sample()
            next_pos = self.next_pos(ghost_pos, d)
            if next_pos.is_valid() and can_move(self.state.ghosts[g], d):
                break

        self.state.ghosts[g].update(next_pos, d)


if __name__ == "__main__":
    env = PocEnv(config)
    env.reset()
    action = env.action_space.sample()
    r = 0
    for i in range(100):
        ob, rw, done, info = env.step(action)
        print(ob)
        print(rw)
        if done:
            break
