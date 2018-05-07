import os
from itertools import zip_longest

import pygame
from gym_pomdp.envs.coord import Coord, Tile

PATH = os.path.split(__file__)[0]
FILE_PATH = os.path.join(PATH, 'assets')


class GuiTile(Tile):
    _borderWidth = 2
    _borderColor = pygame.Color("grey")

    def __init__(self, coord, surface, tile_size=100):
        self.origin = (coord.x * tile_size, coord.y * tile_size)
        self.surface = surface
        self.tile_size = tuple([tile_size] * 2)
        super().__init__(coord)

    def draw(self, img=None, color=pygame.Color("white")):
        rect = pygame.Rect(self.origin, self.tile_size)
        pygame.draw.rect(self.surface, color, rect, 0)  # draw tile
        if img is not None:
            self.surface.blit(img, self.origin)
        pygame.draw.rect(self.surface, GuiTile._borderColor, rect, GuiTile._borderWidth)  # draw border


class GridGui(object):
    _assets = {}

    def __init__(self, x_size, y_size, tile_size):
        self.x_size = x_size
        self.y_size = y_size
        self.n_tiles = x_size * y_size
        self.tile_size = tile_size
        self.assets = {}
        for k, v in self._assets.items():
            self.assets[k] = pygame.transform.scale(pygame.image.load(v), [tile_size] * 2)

        self.w = self.tile_size * self.x_size
        self.h = (self.tile_size * self.y_size) + 50  # size of the taskbar

        pygame.init()
        self.surface = pygame.display.set_mode((self.w, self.h))
        self.surface.fill(pygame.Color("white"))
        self.action_font = pygame.font.SysFont("monospace", 18)
        self.build_board()

    def build_board(self):
        self.board = []
        for idx in range(self.n_tiles):
            tile = GuiTile(self.get_coord(idx), surface=self.surface, tile_size=self.tile_size)
            tile.draw(img=None)
            self.board.append(tile)

    def get_coord(self, idx):
        assert idx >= 0 and idx < self.n_tiles
        return Coord(idx % self.x_size, idx // self.x_size)

    def draw(self, update_board=False):
        raise NotImplementedError()

    def render(self, state, msg):
        raise NotImplementedError()

    def task_bar(self, msg):
        assert msg is not None
        txt = self.action_font.render(msg, 2, pygame.Color("black"))
        rect = pygame.Rect((0, self.h - 50 + 5), (self.w, 40))  # 205 for tiger
        pygame.draw.rect(self.surface, pygame.Color("white"), rect, 0)
        self.surface.blit(txt, (self.tile_size // 2, self.h - 50 + 10))  # 210

    @staticmethod
    def _dispatch():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None


class ShipGui(GridGui):
    _tile_size = 50

    def __init__(self, board_size=(5, 5), obj_pos=()):
        super().__init__(*board_size, tile_size=self._tile_size)

        self.obj_pos = obj_pos
        self.last_shot = -1
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for pos in self.obj_pos:
                self.board[pos].draw(color=pygame.Color("black"))
        else:
            if self.last_shot in self.obj_pos:
                self.board[self.last_shot].draw(color=pygame.Color("red"))
            elif self.last_shot > -1:
                self.board[self.last_shot].draw(color=pygame.Color("blue"))

    def render(self, state, msg):
        self.last_shot = state
        self.draw()
        self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class RockGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _ROBOT=os.path.join(FILE_PATH, "r2d2.png"),
        _ROCK=os.path.join(FILE_PATH, "rock.png")
    )

    def __init__(self, board_size=(5, 5), start_pos=0, obj=()):
        super().__init__(*board_size, tile_size=self._tile_size)
        self.history = [start_pos] * 2
        self.obj = obj
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):

        if update_board:
            for obj in self.obj:
                if obj[1] == -1:
                    color = pygame.Color('red')
                elif obj[1] == 1:
                    color = pygame.Color('blue')
                else:
                    color = pygame.Color('grey')
                self.board[obj[0]].draw(img=self.assets["_ROCK"], color=color)

        last_state = self.history.pop(0)
        if last_state in self.obj:
            self.board[last_state[0]].draw(img=self.assets["_ROCK"])
        else:
            self.board[last_state].draw()

        self.board[self.history[-1]].draw(img=self.assets["_ROBOT"])

    def render(self, state, msg=None):
        self.history.append(state)
        self.draw(update_board=True)
        # self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class TagGui(GridGui):
    _tile_size = 50
    _assets = dict(
        _ROBOT=os.path.join(FILE_PATH, "r2d2.png"),
        _STORM=os.path.join(FILE_PATH, "soldier.png"),
        _RIGHT=os.path.join(FILE_PATH, "right.png"),
        _LEFT=os.path.join(FILE_PATH, "left.png"),
        _UP=os.path.join(FILE_PATH, "up.png"),
        _DOWN=os.path.join(FILE_PATH, "down.png"),

    )

    def __init__(self, board_size=(10, 5), start_pos=0, obj_pos=()):
        super().__init__(*board_size, tile_size=self._tile_size)
        self.obj_pos = obj_pos
        self.agent_history = [start_pos] * 2
        self.update_opp(obj_pos)
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for t in self.board:
                t.draw()

        for opp in self.opp_history:
            old, new = opp
            self.board[old].draw()
            if new is not None:
                self.board[new].draw(img=self.assets["_STORM"])

        self.opp_history = []

        last_state = self.agent_history.pop(0)
        self.board[last_state].draw()
        self.board[self.agent_history[-1]].draw(img=self.assets["_ROBOT"])

    def render(self, state, msg=None):

        agent_pos, obj_pos = state
        self.agent_history.append(agent_pos)
        self.update_opp(obj_pos)
        self.draw(update_board=True)
        if msg is not None:
            self.task_bar(msg)

        pygame.display.update()
        GridGui._dispatch()

    def update_opp(self, opp_pos):
        self.opp_history = []
        for old, new in zip_longest(self.obj_pos, opp_pos):
            self.opp_history.append((old, new))
        self.obj_pos = opp_pos


class TigerGui(GridGui):
    _tile_size = 200
    _assets = dict(
        _CAT=os.path.join(PATH, "assets/cat.png"),
        _DOOR=os.path.join(PATH, "assets/door.png"),
        _BEER=os.path.join(PATH, "assets/beer.png"))

    def __init__(self, board_size=(2, 1)):
        super(TigerGui, self).__init__(*board_size, tile_size=self._tile_size)

        self.draw(update_board=True)
        pygame.display.update()
        TigerGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for tile in self.board:
                tile.draw(img=self.assets["_DOOR"])

    def render(self, state, msg=None):

        if msg is not None:
            self.task_bar(msg)

        action, state = state

        if action < 2 and action == state:
            self.board[state].draw(img=self.assets["_CAT"])
        elif action < 2 and action != state:
            self.board[action].draw(img=self.assets["_BEER"])
        else:
            self.draw(update_board=True)

        pygame.display.update()
        TigerGui._dispatch()
