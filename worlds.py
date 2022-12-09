from enums import GridTile
import numpy as np

world1 = np.full((5,5),GridTile.AIR)
world1[4,4] = GridTile.GOAL

world2 = np.full((5,5),GridTile.AIR)
world2[4,4] = GridTile.GOAL
world2[3,4] = GridTile.DOOR_CLOSED
world2[4,3] = GridTile.DOOR_CLOSED
