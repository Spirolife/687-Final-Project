from enum import Enum
import numpy as np


class GridTile(Enum):
    # AIR
    AIR = 0
    # PLACEABLE
    WALL = 1
    DOOR_CLOSED = 2
    ROCK = 3
    WATER = 4
    # RESERVED (DONT PLACE)
    DOOR_OPEN = 5
    GOAL = 6
    START = 7


class Action(Enum):
    WALK_UP = 0
    WALK_LEFT = 1
    WALK_DOWN = 2
    WALK_RIGHT = 3

    OPEN_UP = 4
    OPEN_LEFT = 5
    OPEN_DOWN = 6
    OPEN_RIGHT = 7

    SWIM_UP = 8
    SWIM_LEFT = 9
    SWIM_DOWN = 10
    SWIM_RIGHT = 11

    JUMP_UP = 12
    JUMP_LEFT = 13
    JUMP_DOWN = 14
    JUMP_RIGHT = 15


class Observation(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


"""
    The following dictionary maps abstract actions from `self.action_space` to 
    the direction we will walk in if that action is taken.
    I.e. 0 corresponds to "up" etc.
"""
action_to_direction = {
    Action.WALK_UP:    np.array([-1, 0]),
    Action.WALK_LEFT:  np.array([0, -1]),
    Action.WALK_DOWN:  np.array([1, 0]),
    Action.WALK_RIGHT: np.array([0, 1]),

    Action.OPEN_UP: np.array([-1, 0]),
    Action.OPEN_LEFT: np.array([0, -1]),
    Action.OPEN_DOWN: np.array([1, 0]),
    Action.OPEN_RIGHT: np.array([0, 1]),

    Action.SWIM_UP: np.array([-1, 0]),
    Action.SWIM_LEFT: np.array([0, -1]),
    Action.SWIM_DOWN: np.array([1, 0]),
    Action.SWIM_RIGHT: np.array([0, 1]),

    Action.JUMP_UP: np.array([-1, 0]),
    Action.JUMP_LEFT: np.array([0, -1]),
    Action.JUMP_DOWN: np.array([1, 0]),
    Action.JUMP_RIGHT: np.array([0, 1]),
}
