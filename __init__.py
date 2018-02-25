from gym.envs.registration import register
register(
    id = "tiger-v0",
    entry_point ="gym_pomdp:Tiger"
)
register(
    id = "tag-v0",
    entry_point ="gym_pomdp:Tag"
)
register(
    id = "battleship-v0",
    entry_point ="gym_pomdp:BattleShip"
)
register(
    id = "rock-v0",
    entry_point ="gym_pomdp:Rock"
)
