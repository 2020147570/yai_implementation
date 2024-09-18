import argparse
from visualize_test import GraphicDisplay
from environment import grid_world
from agent import AGENT


parser = argparse.ArgumentParser()
parser.add_argument('-alg', required=True, help='algorithm type')
args, _ = parser.parse_known_args()


WORLD_HEIGHT = 6
WORLD_WIDTH = 12

env = grid_world(
    HEIGHT=WORLD_HEIGHT,
    WIDTH=WORLD_WIDTH,
    GOAL=[[WORLD_HEIGHT-1, WORLD_WIDTH-1]],
    OBSTACLES=[[WORLD_HEIGHT-1,i] for i in range(1, WORLD_WIDTH-1)]
    )

agent = AGENT(env=env, is_upload=True, alg=args.alg)
grid_world_vis = GraphicDisplay(env=env, agent=agent)
grid_world_vis.print_value_table()
grid_world_vis.mainloop()
