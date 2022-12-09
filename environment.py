# Created by Alenna Spiro and Kobi Falus

import math
import gym
from gym import spaces
import pygame
import numpy as np
from enum import Enum


class GridTile(Enum):
    AIR = 0
    WALL = 1
    DOOR_OPEN = 2
    DOOR_CLOSED = 3
    ROCK = 4
    WATER = 5
    GOAL = 6


class Action(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

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


class OBSERVATION(Enum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "up" etc.
        """
    _action_to_direction = {
        Action.UP:    np.array([-1, 0]),
        Action.LEFT:  np.array([0, -1]),
        Action.DOWN:  np.array([1, 0]),
        Action.RIGHT: np.array([0, 1]),

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
    
    
    # Observations are dictionaries with the state space, in this case the agent's view around itself
    # Each location is encoded as an element of {0, ..., `size`}
    observation_space = spaces.Dict({
        # GridTile
        OBSERVATION.UP:    spaces.Discrete(7),
        OBSERVATION.DOWN:  spaces.Discrete(7),
        OBSERVATION.LEFT:  spaces.Discrete(7),
        OBSERVATION.RIGHT: spaces.Discrete(7)
    })
        
    def __init__(self, render_mode=None, size=5):

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.grid = self.generate_grid(self.size)

        # We have 16 actions
        self.action_space = spaces.Discrete(16)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def _get_obs(self):
        return {
            OBSERVATION.UP: {self.grid[self._agent_location + self._action_to_direction[Action.UP]]},
            OBSERVATION.DOWN: {self.grid[self._agent_location + self._action_to_direction[Action.DOWN]]},
            OBSERVATION.LEFT: {self.grid[self._agent_location + self._action_to_direction[Action.LEFT]]},
            OBSERVATION.RIGHT: {self.grid[self._agent_location + self._action_to_direction[Action.RIGHT]]}
        }

    def generate_grid(self, size):
        grid = np.zeros((size, size))
        wall_count = 4

        door_loc = np.random.randint(0, size, 2)
        swim_loc = np.random.randint(0, size, 2)
        rock_loc = np.random.randint(0, size, 2)

        grid[door_loc] = GridTile.DOOR_CLOSED
        grid[swim_loc] = GridTile.WATER
        grid[rock_loc] = GridTile.ROCK

        for i in range(wall_count):
            grid[np.random.randint(0, size, 2)] = GridTile.WALL

        dist_list = np.full((size, size), math.inf)
        unvisited = np.ones((size, size))
        parent = {}

        q = np.zeros((size, size))
        q[0,0] = 1

        dist_list[0, 0] = 0

        while np.any(q == 1):
            u = np.argmin(dist_list * q)[0]

        return grid

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def get_next_state(self, action):
        if action == Action.OPEN_UP or action == Action.OPEN_DOWN or action == Action.OPEN_LEFT or action == Action.OPEN_RIGHT:
            direction = self._action_to_direction[action]
            if self.grid[self._agent_location + self._action_to_direction[Action.UP]] == GridTile.DOOR_CLOSED:
                self.grid[self._agent_location + self._action_to_direction[Action.UP]] = GridTile.DOOR_OPEN
            return self._agent_location
        if action == Action.OPEN_DOWN:
            if self.grid[self._agent_location + self._action_to_direction[Action.DOWN]] == GridTile.DOOR_CLOSED:
                self.grid[self._agent_location + self._action_to_direction[Action.DOWN]] = GridTile.DOOR_OPEN
            return self._agent_location

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        new_loc = self._agent_location + direction
        # We use `np.clip` to make sure we don't leave the grid
        new_loc = np.clip(new_loc, 0, self.size - 1)
        if self.grid[tuple(new_loc)] == GridTile.WALL:
            # We can't walk into walls
            new_loc = self._agent_location

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    GridWorldEnv()