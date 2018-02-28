import numpy as np
from enum import Enum


class Coord(object):
    def __init__(self, x=0, y=0):
        self.x = x  # rows
        self.y = y  # columns

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Coord(x, y)

    def __mul__(self, other):
        self.x *= other.x
        self.y *= other.y
        return Coord(self.x, self.y)

    def __str__(self):
        return "{},{}".format(self.x, self.y)

    def copy(self):
        return Coord(self.x, self.y)

    def is_valid(self):
        return self.x >= 0 and self.y >= 0
    @property
    def to_list(self):
        return (self.x, self.y)


class Tile(object):
    def __init__(self, coord=(0, 0)):
        self.coord = coord

    def set_value(self, value):
        raise NotImplementedError()


class Grid(object):
    def __init__(self, x_size=10, y_size=5):
        self.x_size = x_size
        self.y_size = y_size
        self.n_tiles = self.x_size * self.y_size

    def __iter__(self):
        return iter(self.board)

    def __setitem__(self, coord, value):
        idx = self.get_index(coord)
        self.board[idx] = value

    def __getitem__(self, coord):
        idx = self.get_index(coord)
        return self.board[idx]

    def build_board(self, Tile):
        self.board = []
        for idx in range(self.n_tiles):
            coord = self.get_coord(idx)
            self.board.append(Tile(coord = coord))

    def set_value(self, value, coord):
        raise NotImplementedError()
        # idx = self.get_index(coord)
        # self.board[idx] = value
        #
    def get_value(self, coord):
        raise NotImplementedError()
        # idx = self.get_index(coord)
        # assert coord == self.board[idx].coord
        # return self.board[idx]

    def get_size(self):
        return (self.x_size, self.y_size)

    def get_index(self, coord):
        # this is the order of the board vector
        return self.x_size * coord.y + coord.x

    def is_inside(self, coord):
        return coord.is_valid() and coord.x < self.x_size and coord.y < self.y_size

    def get_coord(self, idx):
        assert idx >= 0 and idx < self.n_tiles
        return Coord(idx % self.x_size, idx // self.x_size)

    # def sample(self):
    #     return self.get_coord(np.random.randint(self.n_tiles))  # sample over n-1 cells
    def sample(self):
        return self.get_coord(np.random.randint(self.n_tiles))

    @staticmethod
    def opposite(move):
        return (move + 2) % 4

    @staticmethod
    def euclidean_distance(c1, c2):
        return np.linalg.norm(np.subtract(c1.to_list(), c2.to_list()), 1)

    @staticmethod
    def manhattan_distance(c1, c2):
        return np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

    @staticmethod
    def directional_distance(c1, c2, d):
        if d == 0:
            return c1.y - c2.y
        elif d == 1:
            return c1.x - c2.x
        elif d == 2:
            return c2.y - c1.x
        elif d == 3:
            return c2.x - c1.x
        else:
            raise NotImplementedError()


class Moves(Enum):
    UP = Coord(0, 1)
    RIGHT = Coord(1, 0)
    DOWN = Coord(0, -1)
    LEFT = Coord(-1, 0)
    NULL = Coord(0, 0)

    @staticmethod
    def get_coord(idx):
        return list(Moves)[idx].value

    @staticmethod
    def sample():
        return np.random.randint(len(Moves))


from unittest import TestCase


class TestCoord(TestCase):
    assert (Coord(3, 3) + Coord(2, 2)) == Coord(5, 5)
    assert (Coord(5, 2) + Coord(2, 5)) == Coord(7, 7)
    assert (Coord(2, 2) + Moves.UP.value) == Coord(2, 3)
    assert (Coord(2, 2) + Moves.LEFT.value) == Coord(1, 2)
    assert (Coord(2, 2) + Moves.DOWN.value) == Coord(2, 1)
    assert (Coord(2, 2) + Moves.RIGHT.value) == Coord(3, 2)

    grid = Grid()


if __name__ == "__main__":
    TestCoord()
