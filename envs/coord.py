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

    def copy(self):
        return Coord(self.x, self.y)

    def is_valid(self):
        return self.x >= 0 and self.y >= 0
    @property
    def to_list(self):
        return [self.x, self.y]


class Grid(object):
    def __init__(self, x_size=10, y_size=5):
        self.x_size = x_size
        self.y_size = y_size
        self.n_tiles = self.x_size * self.y_size  # obs_cells if obs_cells is not None else y_size * x_size
        self.board = []

    def set_value(self, value):
        self.board.append(value)

    def get_value(self, coord):
        idx = self.get_index(coord)
        return self.board[idx]
    @property
    def get_size(self):
        return (self.x_size, self.y_size)

    def get_index(self, coord):
        return self.x_size * coord.x + coord.y

    def is_inside(self, coord):
        return coord.is_valid() and coord.x < self.x_size and coord.y < self.y_size

    def get_coord(self, idx):
        assert idx >= 0 and idx < self.n_tiles
        return Coord(idx % self.x_size, idx // self.x_size)

    def sample(self):
        return self.get_coord(np.random.randint(self.n_tiles))  # sample over n-1 cells
        # return np.random.randint(self.n_tiles)

    @staticmethod
    def euclidean_distance(c1, c2):
        return np.linalg.norm(np.subtract(c1.to_list(), c2.to_list()), 1)

    @staticmethod
    def manhattan_distance(c1, c2):
        return np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)


class Moves(Enum):
    UP = Coord(0, 1)
    LEFT = Coord(1, 0)
    DOWN = Coord(0, -1)
    RIGHT = Coord(-1, 0)
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
    assert (Coord(2, 2) + Moves.LEFT.value) == Coord(3, 2)
    assert (Coord(2, 2) + Moves.DOWN.value) == Coord(2, 1)
    assert (Coord(2, 2) + Moves.RIGHT.value) == Coord(1, 2)


if __name__ == "__main__":
    TestCoord()
