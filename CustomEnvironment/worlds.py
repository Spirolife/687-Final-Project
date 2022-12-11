from enums import GridTile
import numpy as np

world1 = np.full((5, 5), GridTile.AIR)
world1[4, 4] = GridTile.GOAL

world2 = np.full((5, 5), GridTile.AIR)
world2[4, 4] = GridTile.GOAL
world2[3, 4] = GridTile.DOOR_CLOSED
world2[4, 3] = GridTile.DOOR_CLOSED

world3 = np.full((5, 5), GridTile.AIR)
world3[4, 4] = GridTile.GOAL
world3[3, 4] = GridTile.WALL
world3[3, 3] = GridTile.WALL
world3[3, 2] = GridTile.WALL
world3[3, 1] = GridTile.ROCK
world3[3, 0] = GridTile.WATER
world3[4, 3] = GridTile.DOOR_CLOSED

world4 = np.full((5, 5), GridTile.AIR)
world4[4, 4] = GridTile.GOAL
world4[4, 2] = GridTile.WALL
world4[3, 2] = GridTile.WALL
world4[2, 4] = GridTile.DOOR_CLOSED
world4[2, 3] = GridTile.WALL

world5 = np.full((5, 5), GridTile.AIR)
world5[4, 4] = GridTile.GOAL
world5[1, 0] = GridTile.ROCK
world5[4, 3] = GridTile.WALL

world6 = np.full((7, 7), GridTile.AIR)
world6[6, 6] = GridTile.GOAL
world6[1, 1] = GridTile.WALL
world6[1, 2] = GridTile.WATER
world6[4, 2] = GridTile.WATER
world6[1, 3] = GridTile.WALL
world6[1, 4] = GridTile.WATER
world6[4, 4] = GridTile.ROCK
world6[5, 6] = GridTile.DOOR_CLOSED
world6[6, 5] = GridTile.DOOR_CLOSED
