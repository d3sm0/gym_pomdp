from enum import Enum
import numpy as np
from gym import Env
from gym.spaces import Discrete
from gym_pomdp.envs.coord import Grid, Coord
from gym_pomdp.envs.gui import ShipGui


class Compass(Enum):
    North = Coord(0, 1)
    East = Coord(1, 0)
    South = Coord(0, -1)
    West = Coord(-1, 0)
    Null = Coord(0, 0)
    NorthEast = Coord(1, 1)
    SouthEast = Coord(1, -1)
    SouthWest = Coord(-1, -1)
    NorthWest = Coord(-1, 1)

    @staticmethod
    def get_coord(idx):
        return list(Compass)[idx].value


class Obs(Enum):
    NULL = 0
    HIT = 1


class Ship(object):
    def __init__(self, coord, length):
        self.pos = coord
        self.direction = np.random.randint(4)
        self.length = length


class ShipState(object):
    def __init__(self):
        self.ships = []
        self.total_remaining = 0


class Cell(object):
    def __init__(self):
        self.occupied = False
        self.visited = False
        self.diagonal = False


class BattleGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)

    def build_board(self, value=0):
        self.board = []
        for idx in range(self.n_tiles):
            self.board.append(Cell())
        self.board = np.asarray(self.board).reshape((self.x_size, self.y_size))


class BattleShipEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=(5, 5), max_len=3):
        self.grid = BattleGrid(board_size)
        self.action_space = Discrete(self.grid.n_tiles)
        self.observation_space = Discrete(len(Obs))
        self.num_obs = 2
        self.rw_range = self.action_space.n / 4.
        self.discount = 1.
        self.total_remaining = max_len - 1
        self.max_len = max_len + 1

    def _step(self, action):

        assert self.done == False
        assert self.action_space.contains(action)
        self.last_action = action
        self.t += 1
        action_pos = self.grid.get_coord(action)
        # cell = self.grid.get_value(action_pos)
        cell = self.grid[action_pos]
        reward = 0
        if cell.visited:
            reward -= 10
            obs = Obs.NULL.value
        else:
            if cell.occupied:
                reward -= 1
                obs = 1
                self.bsstate.total_remaining -= 1

                for d in range(4, 8):
                    if self.grid[action_pos + Compass.get_coord(d)]:
                        self.grid[action_pos + Compass.get_coord(d)].diagonal = False
            else:
                reward -= 1
                obs = Obs.NULL.value
            cell.visited = True
        if self.bsstate.total_remaining == 0:
            reward += self.grid.n_tiles
            self.done = True
        self.tot_rw += reward
        return obs, reward, self.done, {"state": self.bsstate}

    def _reset(self):
        self.done = False
        self.tot_rw = 0
        self.t = 0
        self.last_action = -1
        self._init_state()
        return Obs.NULL.value

    def _render(self, mode='human', close=False):
        if mode == 'human':
            if not hasattr(self, "gui"):
                obj_pos = []
                for ship in self.bsstate.ships:
                    pos = ship.pos
                    obj_pos.append(self.grid.get_index(pos))
                    for i in range(ship.length):
                        pos += Compass.get_coord(ship.direction)
                        obj_pos.append(self.grid.get_index(pos))
                self.gui = ShipGui(board_size=self.grid.get_size(), obj_pos=obj_pos)
            if self.t > 0:
                msg = "A: " + str(self.grid.get_coord(self.last_action)) + "T: " + str(self.t) + "Rw :" + str(
                    self.tot_rw)
                self.gui.render(state=self.last_action, msg=msg)

    def _close(self):
        pass

    def _init_state(self):
        self.bsstate = ShipState()
        self.grid.build_board()

        for length in reversed(range(2, self.max_len)):
            # num_ships = 1
            # for idx in range(num_ships):
            while True:  # add one ship of each kind
                ship = Ship(coord=self.grid.sample(), length=length)
                if not self.collision(ship):
                    break
            self.mark_ship(ship)
            self.bsstate.ships.append(ship)

    def mark_ship(self, ship):

        pos = ship.pos  # .copy()

        for i in range(ship.length):
            cell = self.grid[pos]
            assert not cell.occupied
            cell.occupied = True
            if not cell.visited:
                self.bsstate.total_remaining += 1
            pos += Compass.get_coord(ship.direction)

    def collision(self, ship):

        pos = ship.pos  # .copy()
        for i in range(ship.length):
            if not self.grid.is_inside(pos + Compass.get_coord(ship.direction)):
                return True
            # cell = self.grid.get_value(pos)
            cell = self.grid[pos]
            if cell.occupied:
                return True
            for adj in range(8):
                coord = pos + Compass.get_coord(adj)
                if self.grid.is_inside(coord) and self.grid[coord].occupied:
                    return True
            pos += Compass.get_coord(ship.direction)
        return False


if __name__ == "__main__":
    env = BattleShipEnv()
    ob = env.reset()
    env.render()
    done = False
    t = 0
    while not done:
        action = env.action_space.sample()
        ob, rw, done, info = env.step(action)
        env.render()
        t += 1
    env.close()

    print("rw {}, t{}".format(rw, t))
