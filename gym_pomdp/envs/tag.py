import numpy as np
from gym import Env
from gym.spaces import Discrete

from gym_pomdp.envs.coord import Coord, Moves, Grid
from gym_pomdp.envs.gui import TagGui


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
        return "stay"
    else:
        raise NotImplementedError()


class TagGrid(Grid):
    def __init__(self, board_size, obs_cells=29):
        super().__init__(*board_size)
        self.build_board(0)
        self.n_tiles = obs_cells

    def is_inside(self, coord):
        # return coord.is_valid() and coord.x < self.x_size and coord.y < self.y_size
        if coord.y >= 2:
            return coord.x >= 5 and coord.x < 8 and coord.y < 5
        else:
            return coord.x >= 0 and coord.x < 10 and coord.y >= 0

    def get_coord(self, idx):
        if idx == -1:
            return Coord(-1, -1)
        # return Coord(idx % self.x_size, idx // self.x_size)
        # assert index >= 0 and index < self.n_tiles
        if idx < 20:
            return Coord(idx % 10, idx // 10)
        # if idx == -1:
        #     return Coord(-1, -1) # out of the board
        idx -= 20
        return Coord(idx % 3 + 5, idx // 3 + 2)

    def get_index(self, coord):
        if coord.y == -1:
            return -1
        # return self.x_size * coord.y + coord.x
        # assert coord.x >= 0 and coord.x < 10
        # assert coord.y >= 0 and coord.y < 5
        if coord.y < 2:
            return coord.y * 10 + coord.x
        # assert coord.x >= 5 and coord.x < 8
        return 20 + (coord.y - 2 * 3) + coord.x - 5


grid = TagGrid((10, 5), obs_cells=29)


class TagEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, board_size=(10, 5), num_opponents=4, obs_cells=29, move_prob=.8):
        self.num_opponents = num_opponents
        self.move_prob = move_prob
        self._reward_range = 10 * self.num_opponents
        self._discount = .95
        self.action_space = Discrete(len(Moves))
        self.grid = TagGrid(board_size, obs_cells=obs_cells)
        self.observation_space = Discrete(self.grid.n_tiles + 1)
        self.time = 0

    def reset(self):
        self.done = False
        self.time = 0
        self.last_action = 4
        self.state = self._get_init_state(False)
        return self._sample_ob(state=self.state, action=0)  # get agent position

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def step(self, action):  # state
        assert self.action_space.contains(action)  # action are idx of movements
        # assert self.done == False

        reward = 0.
        self.time += 1
        self.last_action = action
        if action == 4:
            tagged = False
            for opp, opp_pos in enumerate(self.state.opponent_pos):
                if opp_pos == self.state.agent_pos:  # check if x==x_agent and y==y_agent
                    reward = 10.
                    tagged = True
                    self.state.opponent_pos[opp] = Coord(-1, -1)
                    self.state.num_opp -= 1
                    # self.state.opponent_pos.pop(opp)
                elif self.state.opponent_pos[opp].is_valid():
                    self.move_opponent(opp)
            if not tagged:
                reward = -10.

        else:
            reward = -1.
            next_pos = self.state.agent_pos + Moves.get_coord(action)
            if self.grid.is_inside(next_pos):
                self.state.agent_pos = next_pos

        ob = self._sample_ob(self.state, action)
        p_ob = self._compute_prob(action, self.state, ob)
        done = self.state.num_opp == 0
        return ob, reward, done, {"state": self._encode_state(self.state), "p_ob": p_ob}

    def render(self, mode="human", close=False):
        if close:
            return
        if mode == "human":
            agent_pos = self.grid.get_index(self.state.agent_pos)
            obj_pos = [self.grid.get_index(opp) for opp in self.state.opponent_pos]
            if not hasattr(self, "gui"):
                self.gui = TagGui(board_size=self.grid.get_size, start_pos=agent_pos, obj_pos=obj_pos)
            msg = "S: " + str(self.state) + " T: " + str(self.time) + " A: " + action_to_str(
                self.last_action)
            self.gui.render(state=(agent_pos, obj_pos), msg=msg)

    def _encode_state(self, state):
        s = np.zeros(self.num_opponents + 1, dtype=np.int32) - 1
        s[0] = grid.get_index(state.agent_pos)
        for idx, opp in zip(range(1, len(s)), state.opponent_pos):
            opp_idx = grid.get_index(opp)
            s[idx] = opp_idx

        return s

    def _decode_state(self, state):
        agent_idx = state[0]
        tag_state = TagState(grid.get_coord(agent_idx))
        for opp_idx in state[1:]:
            if opp_idx > -1:
                tag_state.num_opp += 1
            opp_pos = grid.get_coord(opp_idx)
            tag_state.opponent_pos.append(opp_pos)

            # true_pos = [grid.get_index(pos) for pos in env.state.opponent_pos]
        # assert np.all(state[1:] == true_pos)
        # assert np.all(tag_state.opponent_pos == env.state.opponent_pos)
        return tag_state

    def _get_init_state(self, should_encode=True):

        tag_state = TagState(self.grid.sample())
        for opp in range(self.num_opponents):
            tag_state.opponent_pos.append(self.grid.sample())

        tag_state.num_opp = self.num_opponents
        assert len(tag_state.opponent_pos) > 0
        return tag_state if not should_encode else self._encode_state(tag_state)

    def _set_state(self, state):
        # self.reset()
        self.done = False
        self.state = self._decode_state(state)

    def move_opponent(self, opp):
        opp_pos = self.state.opponent_pos[opp]
        actions = self._admissable_actions(self.state.agent_pos, opp_pos)
        if np.random.binomial(1, self.move_prob):
            move = np.random.choice(actions).value
            if self.grid.is_inside(opp_pos + move):
                self.state.opponent_pos[opp] = opp_pos + move

    def _compute_prob(self, action, next_state, ob, correct_prob=.85):
        return int(ob == grid.get_index(next_state.agent_pos))

    def _sample_ob(self, state, action):
        obs = grid.get_index(state.agent_pos)  # agent index
        if action < 4:
            for opp_pos in state.opponent_pos:
                if opp_pos == state.agent_pos:
                    obs = grid.n_tiles  # number of cells that can observe
        assert 0 <= obs <= 30
        return obs

    def _generate_legal(self):
        return list(range(self.action_space.n))

    def _local_move(self, state, last_action, last_ob):
        if len(state.opponent_pos) > 0:
            opp = np.random.randint(len(state.opponent_pos))
        else:
            return False

        if state.opponent_pos[opp] == Coord(-1, -1):
            return False
        state.opponent_pos[opp] = self.grid.sample()
        if last_ob != self.grid.get_index(state.agent_pos):
            state.agent_pos = self.grid.get_coord(last_ob)

        ob = self._sample_ob(state, last_action)
        return ob == last_ob

    @staticmethod
    def _admissable_actions(agent_pos, opp_pos):
        actions = []
        if opp_pos.x >= agent_pos.x:
            actions.append(Moves.RIGHT)
        if opp_pos.y >= agent_pos.y:
            actions.append(Moves.UP)
        if opp_pos.x <= agent_pos.x:
            actions.append(Moves.LEFT)
        if opp_pos.y <= agent_pos.y:
            actions.append(Moves.DOWN)
        if opp_pos.x == agent_pos.x and opp_pos.y > agent_pos.y:
            actions.append(Moves.UP)
        if opp_pos.y == agent_pos.y and opp_pos.x > agent_pos.x:
            actions.append(Moves.RIGHT)
        if opp_pos.x == agent_pos.x and opp_pos.y < agent_pos.y:
            actions.append(Moves.DOWN)
        if opp_pos.y == agent_pos.y and opp_pos.x < agent_pos.x:
            actions.append(Moves.LEFT)
        assert len(actions) > 0
        return actions


class TagState(object):
    def __init__(self, coord):
        self.agent_pos = coord
        self.opponent_pos = []
        self.num_opp = 0

    def __str__(self):
        return str(len(self.opponent_pos))


if __name__ == '__main__':
    env = TagEnv()
    # env.reset()
    # state = env.tag_state
    # gui = RobotGrid(start_coord=state.agent_pos, obj_coord=state.opponent_pos)
    # action = Action.sample()
    # env.step(action=action)
    # gui.render(action, env.tag_state)
    #
    ob = env.reset()
    env.render()
    done = False
    r = 0
    while not done:
        action = Moves.sample()
        ob1, rw, done, info = env.step(action)
        env._set_state(info['state'])
        env.render()
        r += rw
    print('done, r {}'.format(r))
