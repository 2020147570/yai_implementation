from environment import grid_world
from agent import AGENT


WORLD_HEIGHT = 6
WORLD_WIDTH = 12

env = grid_world(
    HEIGHT=WORLD_HEIGHT,
    WIDTH=WORLD_WIDTH,
    GOAL=[[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
    OBSTACLES=[[WORLD_HEIGHT-1,i] for i in range(1, WORLD_WIDTH-1)]
    )

agent = AGENT(env=env, is_upload=False)
agent.Q_learning(epsilon=0.4, decay_period=10000, decay_rate=0.8)
