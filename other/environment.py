# Created by Alenna Spiro and Kobi Falus

import math
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from enums import GridTile, Action, Observation, action_to_direction
from worlds import world1, world2


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Observations are dictionaries with the state space, in this case the agent's view around itself
    # Each location is encoded as an element of {0, ..., `size`}
    observation_space = spaces.Dict({
        # GridTile
        Observation.UP:    spaces.Discrete(7),
        Observation.DOWN:  spaces.Discrete(7),
        Observation.LEFT:  spaces.Discrete(7),
        Observation.RIGHT: spaces.Discrete(7)
    })

    def __init__(self, world=None, render_mode=None, size=5):

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        if world is None:
            val, grid = self.generate_grid(self.size)
            while (val == False):
                val = self.generate_grid(self.size)
            self.grid = grid
        else:
            self.grid = world

        self.print_grid(self.grid)

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

        self.reset()

    def _get_obs(self):
        cur = self._agent_location
        next_states = np.array([cur + action_to_direction[Action.WALK_UP], cur + action_to_direction[Action.WALK_LEFT],
                                cur + action_to_direction[Action.WALK_DOWN], cur + action_to_direction[Action.WALK_RIGHT]])

        obj_contents = []
        for i, n in enumerate(next_states):
            if n[0] < 0 or n[1] < 0 or n[0] >= self.size or n[1] >= self.size:
                obj_contents.append(GridTile.WALL)
            else:
                obj_contents.append(self.grid[tuple(next_states[i])])

        return {
            Observation.UP: {obj_contents[0]},
            Observation.LEFT: {obj_contents[1]},
            Observation.LEFT: {obj_contents[2]},
            Observation.RIGHT: {obj_contents[3]}
        }

    def generate_grid(self, size):
        grid = np.full((size, size), GridTile.AIR, dtype=GridTile)
        wall_count = 4

        door_loc = np.random.randint(0, size, 2)
        swim_loc = np.random.randint(0, size, 2)
        rock_loc = np.random.randint(0, size, 2)

        grid[tuple(door_loc)] = GridTile.DOOR_CLOSED
        grid[tuple(swim_loc)] = GridTile.WATER
        grid[tuple(rock_loc)] = GridTile.ROCK

        for i in range(wall_count):
            grid[tuple(np.random.randint(0, size, 2))] = GridTile.WALL

        visited = [(0, 0)]
        stack = [(0, 0)]
        parent = {}

        while len(stack) != 0:
            u = stack.pop()
            neighbors = np.array(
                [[u[0]+1, u[1]], [u[0]-1, u[1]], [u[0], u[1]+1], [u[0], u[1]-1]])
            for n in neighbors:
                if n[0] >= 0 and n[1] >= 0 and n[0] < size and n[1] < size and grid[tuple(n)] != GridTile.WALL and tuple(n) not in visited:
                    visited.append(tuple(n))
                    parent[tuple(n)] = u
                    stack.append(tuple(n))

        cur = (4, 4)
        visited = []
        if cur not in visited:
            return False, grid
        while cur != (0, 0):
            cur = parent[cur[0], cur[1]]
            visited.append(cur)
            if len(visited) > size**2:
                return False, grid

        return True, grid

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
        info = None

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = action_to_direction[action]
        reward = -1

        # Determine the new location (check for walls on bounds)
        new_loc = self._agent_location+direction
        clipped_loc = np.clip(new_loc, 0, self.size-1)
        new_cell = self.grid[tuple(clipped_loc)]
        if not np.array_equal(new_loc, clipped_loc):
            new_cell = GridTile.WALL
        new_loc = clipped_loc

        # If open, don't move and open the door
        if action == Action.OPEN_UP or action == Action.OPEN_DOWN or action == Action.OPEN_LEFT or action == Action.OPEN_RIGHT:
            new_loc = self._agent_location
            if new_cell == GridTile.DOOR_CLOSED:
                new_cell = GridTile.DOOR_OPEN
                reward = -0.5
            else:
                reward = -3
        # If jump, move unless blocked by door or wall
        elif action == Action.JUMP_UP or action == Action.JUMP_DOWN or action == Action.JUMP_LEFT or action == Action.JUMP_RIGHT:
            if new_cell == GridTile.ROCK:
                reward = -0.5
            else:
                reward = -3
                if new_cell == GridTile.DOOR_CLOSED or new_cell == GridTile.WALL:
                    new_loc = self._agent_location
        # If swim, move if into water, otherwise don't move
        elif action == Action.SWIM_UP or action == Action.SWIM_DOWN or action == Action.SWIM_LEFT or action == Action.SWIM_RIGHT:
            if new_cell == GridTile.WATER:
                reward = -0.5
            else:
                reward = -3
                new_loc = self._agent_location
        # If walk, move unless blocked by door or wall. Higher penalty for water
        elif action == Action.WALK_UP or action == Action.WALK_DOWN or action == Action.WALK_LEFT or action == Action.WALK_RIGHT:
            if new_cell == GridTile.WALL or new_cell == GridTile.DOOR_CLOSED or new_cell == GridTile.ROCK:
                new_loc = self._agent_location
                reward = -3
            elif new_cell == GridTile.WATER:
                reward = -2

        # Update the location of the agent
        self._agent_location = new_loc

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self._agent_location, self._target_location)
        observation = self._get_obs()
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, None

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

    def print_grid(self, grid):
        print("---------------------")
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                print("| " + str(grid[x, y].value) + " ", end="")
            print("|")
            print("---------------------")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = GridWorldEnv(world1)
    print(env._agent_location)
    env.step(Action.WALK_DOWN)
    print(env._agent_location)
    env.step(Action.WALK_DOWN)
    print(env._agent_location)

    env.close()
