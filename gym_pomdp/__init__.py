from gym.envs.registration import register

register(
    id="Tiger-v0",
    max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TigerEnv"
)
register(
    id="Tag-v0",
    max_episode_steps = 400,
    entry_point="gym_pomdp.envs:TagEnv"
)
register(
    id="Battleship-v0",
    max_episode_steps = 400,
    entry_point="gym_pomdp.envs:BattleShipEnv"
)
register(
    id="Rock-v0",
    max_episode_steps = 400,
    entry_point="gym_pomdp.envs:RockEnv"
)
