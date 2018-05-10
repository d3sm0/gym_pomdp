import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="Tiger-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TigerEnv"
)
register(
    id="Pocman-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:PocEnv"
)
register(
    id="Tag-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TagEnv"
)
register(
    id="Battleship-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:BattleShipEnv"
)
register(
    id="Rock-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:RockEnv"
)
register(
    id="StochasticRock-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:StochasticRockEvn"
)
register(
    id="Network-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:NetworkEnv"
)
register(
    id="Test-v0",
    # max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TestEnv"
)
