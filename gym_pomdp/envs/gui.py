import os
import pygame
from queue import deque

PATH = os.path.split(__file__)[0]


# TODO Check abs path
class Tile(object):
    borderColor = pygame.Color("grey")
    borderWidth = 2

    def __init__(self, x, y, surface, tile_size=(100, 100)):
        self.origin = (x, y)
        self.coord = (x // tile_size[0], y // tile_size[1])
        self.surface = surface
        self.tile_size = tile_size

    def draw(self, img=None, color=pygame.Color("white")):
        rect = pygame.Rect(self.origin, self.tile_size)
        pygame.draw.rect(self.surface, color, rect, 0)
        if img is not None:
            self.surface.blit(img, self.origin)
        pygame.draw.rect(self.surface, Tile.borderColor, rect, Tile.borderWidth)


class GridGui(object):
    tile_size = (100, 100)
    _assets = {}

    def __init__(self, board_size=(2, 2)):
        self.board_size = board_size
        self.assets = {}
        for k, v in self._assets.items():
            self.assets[k] = pygame.transform.scale(pygame.image.load(v), self.tile_size)

        self.w = self.tile_size[0] * board_size[1]
        self.h = self.tile_size[0] * board_size[0] + 50  # size of the taskbar

        pygame.init()
        self.surface = pygame.display.set_mode((self.w, self.h))
        self.surface.fill(pygame.Color("white"))
        self.action_font = pygame.font.SysFont("monospace", 18)
        self.create_tiles()

    def create_tiles(self):
        self.board = []

        for r in range(self.board_size[0]):
            for c in range(self.board_size[1]):
                x = c * self.tile_size[0]
                y = r * self.tile_size[1]
                tile = Tile(x, y, self.surface, tile_size=self.tile_size)
                tile.draw(img=None)
                self.board.append(tile)
                # pygame.display.update()
                # self._dispatch()

        # for idx in range(self.board_size[0] * self.board_size[1]):
        #     x = (idx % self.w) * self.tile_size[0]
        #     y = (idx // self.h) * self.tile_size[1]
        #     tile = Tile(x, y, self.surface, tile_size=self.tile_size)
        #     self.board.append(tile)

    def draw(self, update_board=False):
        raise NotImplementedError()

    def render(self, state, msg):
        raise NotImplementedError()

    def task_bar(self, msg):
        assert msg is not None
        txt = self.action_font.render(msg, 2, pygame.Color("black"))
        rect = pygame.Rect((0, self.h - 50 + 5), (self.w, 40))  # 205 for tiger
        pygame.draw.rect(self.surface, pygame.Color("white"), rect, 0)
        self.surface.blit(txt, (self.tile_size[0] / 2, self.h - 50 + 10))  # 210

    @staticmethod
    def _dispatch():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None


class ShipGui(GridGui):
    tile_size = (50, 50)

    def __init__(self, board_size=(5, 5), start_pos=0, obj_pos=()):
        super().__init__(board_size)
        self.obj_pos = obj_pos
        self.history = deque(maxlen=1)
        self.history.append(start_pos)
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for pos in self.obj_pos:
                self.board[pos].draw(color=pygame.Color("blue"))

        last_shot = self.history.pop()
        if last_shot in self.obj_pos:
            self.board[last_shot].draw(color=pygame.Color("red"))

    def render(self, state, msg):
        self.history.append(state)
        self.draw()
        self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class RockGui(GridGui):
    tile_size = (50, 50)
    _assets = dict(
        _ROBOT=os.path.abspath("assets/r2d2.png"),
        _ROCK=os.path.abspath("assets/rock.png")
    )

    def __init__(self, board_size=(5, 5), start_pos=0, obj_pos=()):
        super().__init__(board_size=board_size)
        self.history = deque(maxlen=2)
        self.history.append(start_pos)
        # self.agent_pos = start_coord
        self.obj_pos = obj_pos
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):

        for idx, pos in enumerate(self.history):
            if idx < self.history.maxlen - 1:
                if pos in self.obj_pos:
                    self.board[pos].draw(img=self.assets["_ROCK"])
                else:
                    self.board[pos].draw()
            else:
                self.board[pos].draw(img=self.assets["_ROBOT"])
        if update_board:
            for obj in self.obj_pos:
                self.board[obj].draw(img=self.assets["_ROCK"])

    def render(self, state, msg=None):
        self.history.append(state)
        self.draw()
        self.task_bar(msg)
        pygame.display.update()
        GridGui._dispatch()


class TagGui(GridGui):
    tile_size = (50, 50)
    _assets = dict(
        _ROBOT=os.path.abspath("assets/r2d2.png"),
        _STORM=os.path.abspath("assets/soldier.png"),
        _RIGHT=os.path.abspath("assets/right.png"),
        _LEFT=os.path.abspath("assets/left.png"),
        _UP=os.path.abspath("assets/up.png"),
        _DOWN=os.path.abspath("assets/down.png"),

    )

    def __init__(self, board_size=(10, 5), start_pos=0, obj_pos=()):
        super().__init__(board_size=board_size)
        # self.agent_pos = start_coord
        # self.obj_pos = obj
        self.history = deque(maxlen=len(obj_pos) * 2 + 2)
        self.history.append(start_pos)
        for obj in obj_pos:
            self.history.append(obj)
        self.draw(update_board=True)
        pygame.display.update()
        GridGui._dispatch()

    def draw(self, update_board=False):
        if update_board:
            for t in self.board:
                t.draw()
        agent_idx = (self.history.maxlen) // 2
        for idx, pos in enumerate(self.history):
            if idx == agent_idx and idx > 0:
                self.board[pos].draw(img=self.assets["_ROBOT"])
            elif idx > agent_idx:
                self.board[pos].draw(img=self.assets["_STORM"])
            else:
                self.board[pos].draw()

    def render(self, state, msg=None):

        if len(state) - 1 < (self.history.maxlen - 2) // 2:
            self.history = deque(self.history, maxlen=self.history.maxlen - 2)
        for s in state:
            self.history.append(s)
        self.draw()
        if msg is not None:
            self.task_bar(msg)

        pygame.display.update()
        GridGui._dispatch()


class TigerGui(GridGui):
    tile_size = (200, 200)
    _assets = dict(
        _CAT=os.path.join(PATH, "assets/cat.png"),
        _DOOR=os.path.join(PATH, "assets/door.png"),
        _BEER=os.path.join(PATH, "assets/beer.png"))

    def __init__(self, board_size=(1, 2)):
        super(TigerGui, self).__init__(board_size=board_size)

        self.draw(update_board=True)
        pygame.display.update()
        TigerGui._dispatch()

    def draw(self, update_board=False):
        # self.surface.fill(BLACK)
        if update_board:
            for tile in self.board:
                tile.draw(img=self.assets["_DOOR"])
        # for row in self.board:
        #     for tile in row:
        #         tile.draw(img=self.assets["_DOOR"])

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
