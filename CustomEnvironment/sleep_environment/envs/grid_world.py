# Created by Alenna Spiro and Kobi Falus

import math
import gym
from gym import spaces
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
        0:    spaces.Discrete(7),
        1:  spaces.Discrete(7),
        2:  spaces.Discrete(7),
        3: spaces.Discrete(7)
    })

    action_space = spaces.Discrete(16)
    window_size = 512  # The size of the PyGame window

    """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
    """
    window = None
    clock = None

    def __init__(self, world=None, render_mode=None, size=5):

        self.np_random, seed = gym.utils.seeding.np_random()

        self.preset_world = world

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.reset()
        self.print_grid(self.grid)

    def set_world(self, world=None):
        self.preset_world = world
        self.reset()
        # self.print_grid(self.grid)

    def grid_to_nums(self, grid):
        flat = grid.flatten()
        nums = np.array([i.value for i in flat])
        return nums.reshape(grid.shape)

    def _get_obs(self):
        cur = self._agent_location
        next_states = np.array([cur + action_to_direction[Action.WALK_UP], cur + action_to_direction[Action.WALK_LEFT],
                                cur + action_to_direction[Action.WALK_DOWN], cur + action_to_direction[Action.WALK_RIGHT]])
        obj_contents = [GridTile.WALL.value if n[0] < 0 or n[1] < 0 or n[0] >=
                        self.grid.shape[0] or n[1] >= self.grid.shape[1] else self.grid[tuple(n)].value
                        for n in next_states]
        return tuple(obj_contents)

    def find_valid_path(self, grid, start):
        stack = [start]
        visited = set()
        while stack:
            top = stack.pop()
            neighbors = [(top[0]+1, top[1]), (top[0]-1, top[1]),
                         (top[0], top[1]+1), (top[0], top[1]-1)]
            for neighbor in neighbors:
                if neighbor[0] >= 0 and neighbor[0] < grid.shape[0] and \
                   neighbor[1] >= 0 and neighbor[1] < grid.shape[1] and \
                   neighbor not in visited:
                    if grid[neighbor] == GridTile.GOAL:
                        return True
                    elif grid[neighbor] != GridTile.WALL:
                        stack.append(neighbor)
            visited.add(top)
        return False

    def is_grid_valid(self, grid):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == GridTile.WALL or grid[i, j] == GridTile.GOAL:
                    continue
                is_valid = self.find_valid_path(grid, (i, j))
                if not is_valid:
                    return False
        return True

    def generate_grid(self, size):
        grid = np.full((size, size), GridTile.AIR, dtype=GridTile)
        grid[size-1, size-1] = GridTile.GOAL
        been_validated = False

        while not been_validated:
            cur_grid = grid.copy()
            # Generate random things
            for _ in range(self.RAND_SIZE*2):
                cell = tuple(np.random.randint(0, size, 2))
                if cur_grid[cell] == GridTile.AIR:
                    cur_grid[cell] = GridTile(np.random.randint(1, 5))
            been_validated = self.is_grid_valid(cur_grid)

        return cur_grid

    RAND_SIZE = 5

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.grid = self.preset_world if self.preset_world is not None else self.generate_grid(
            self.RAND_SIZE)
        self.preset_world = self.grid.copy()

        # Get goal location
        goal_loc = np.argwhere(self.grid == GridTile.GOAL).flatten()
        if goal_loc.shape[0] == 0:
            goal_loc = np.random.randint(0, self.grid.shape, 2)
            self.grid[tuple(goal_loc)] = GridTile.GOAL
        self._target_location = goal_loc

        start = np.argwhere(self.grid == GridTile.START).flatten()
        if start.shape[0] == 2:
            self.grid[tuple(start)] = GridTile.AIR
        else:
            start = np.random.randint(0, self.grid.shape, 2)
            while self.grid[tuple(start)] == GridTile.GOAL or self.grid[tuple(start)] == GridTile.WALL:
                start = np.random.randint(0, self.grid.shape, 2)
        self._agent_location = start
        self._start = start

        observation = self._get_obs()
        info = None

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action_num):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = Action(action_num)
        direction = action_to_direction[action]
        reward = -1

        # Determine the new location (check for walls on bounds)
        new_loc = self._agent_location+direction
        clipped_loc = np.clip(new_loc, 0, np.array(self.grid.shape)-1)
        new_cell = self.grid[tuple(clipped_loc)]
        if not np.array_equal(new_loc, clipped_loc):
            new_cell = GridTile.WALL
        new_loc = clipped_loc

        # If open, don't move and open the door
        if action == Action.OPEN_UP or action == Action.OPEN_DOWN or action == Action.OPEN_LEFT or action == Action.OPEN_RIGHT:
            new_loc = self._agent_location
            if new_cell == GridTile.DOOR_CLOSED:
                self.grid[tuple(clipped_loc)] = GridTile.DOOR_OPEN
                reward = -1
            else:
                reward = -10
        # If jump, move unless blocked by door or wall
        elif action == Action.JUMP_UP or action == Action.JUMP_DOWN or action == Action.JUMP_LEFT or action == Action.JUMP_RIGHT:
            if new_cell == GridTile.ROCK:
                reward = -1
            else:
                reward = -10
                if new_cell == GridTile.DOOR_CLOSED or new_cell == GridTile.WALL or new_cell == GridTile.GOAL:
                    new_loc = self._agent_location
        # If swim, move if into water, otherwise don't move
        elif action == Action.SWIM_UP or action == Action.SWIM_DOWN or action == Action.SWIM_LEFT or action == Action.SWIM_RIGHT:
            if new_cell == GridTile.WATER:
                reward = -1
            else:
                reward = -10
                new_loc = self._agent_location
        # If walk, move unless blocked by door or wall. Higher penalty for water
        elif action == Action.WALK_UP or action == Action.WALK_DOWN or action == Action.WALK_LEFT or action == Action.WALK_RIGHT:
            if new_cell == GridTile.WALL or new_cell == GridTile.DOOR_CLOSED or new_cell == GridTile.ROCK:
                new_loc = self._agent_location
                reward = -10
            elif new_cell == GridTile.WATER:
                reward = -10
            elif new_cell == GridTile.GOAL:
                reward = 0
        else:
            raise ValueError("Invalid action")

        # Update the location of the agent
        self._agent_location = new_loc

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(
            self._agent_location, self._target_location)
        observation = self._get_obs()
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        if terminated:
            reward = 0

        return observation, reward, terminated, None

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

    def print_grid(self, grid, plot_agent=True):
        grid_nums = self.grid_to_nums(grid).astype(str)
        if plot_agent:
            grid_nums[tuple(self._agent_location)] = 'X'
        for row in grid_nums:
            print(" ".join([str(cell) for cell in row]))
        print()

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
