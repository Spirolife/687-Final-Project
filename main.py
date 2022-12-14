import argparse
from tqdm import tqdm
import numpy as np
import gym
import sleep_environment
import CustomEnvironment.worlds as worlds
from enums import GridTile, Action, Observation, action_to_direction
from scipy.special import softmax
# observation, info = env.reset()

# for _ in range(1000):
#     # agent policy that uses the observation and info
#     action = env.action_space.sample()
#     # print(action)
#     observation, reward, terminated, info = env.step(action)
#     # print(observation, reward, terminated, info)

#     if terminated:
#         observation, info = env.reset()

# env.close()


class Agent:
    def __init__(self, world=None):
        self.env = gym.make("GridWorldEnv-v0", world=world)
        self.Q = np.zeros((7, 7, 7, 7, 16))
        self.V = np.zeros((7, 7, 7, 7))
        self.set_rand_size(4)

        # For Actor-Critic
        # self.AC_P = np.zeros(())

    TIMEOUT = 1000

    def q_learn(self, episodes=3000, gamma=1, alpha=0.1, epsilon='decay'):
        if epsilon == 'decay':
            def epsilon_fn(x): return 1 / (x + 1)
        else:
            def epsilon_fn(_): return epsilon
        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            done = False
            count = 0
            while not done and count < self.TIMEOUT:
                count += 1
                if np.random.random() < epsilon_fn(episode):
                    action = self.env.action_space.sample()
                else:
                    action = np.random.choice(
                        np.arange(16), p=softmax(self.Q[state]))

                next_state, reward, done, _ = self.env.step(action)

                self.Q[state][action] = self.Q[state][action] + alpha * \
                    (reward + gamma *
                     np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state

    def monte_carlo(self, episodes=1000):
        returns = {}
        for episode in tqdm(range(episodes)):
            # timesteps x [state action reward]
            seen_sa = []
            seen_s = []
            trajectory = []
            state, info = self.env.reset()
            # print(state)
            done = False
            count = 0
            trajectory.append([state, None, None])

            while not done and count < self.TIMEOUT:
                count += 1
                action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append([next_state, action, reward])

            # print(returns)

            return_G = trajectory[-1][2]
            for t in range(len(trajectory)-2, 0, -1):
                if (trajectory[t][0], trajectory[t+1][1]) not in seen_sa:
                    seen_sa.append((trajectory[t][0], trajectory[t+1][1]))
                    if trajectory[t][0] in returns:
                        returns[trajectory[t][0], trajectory[t+1][1]].append(return_G)
                    else:
                        returns[trajectory[t][0], trajectory[t+1][1]] = [return_G]
                    self.Q[trajectory[t][0]][trajectory[t+1][1]] = np.average(returns[trajectory[t][0], trajectory[t+1][1]])
                if trajectory[t][0] not in seen_s:
                    seen_s.append(trajectory[t][0])
                return_G += trajectory[t][2]

    def sarsa(self, episodes=1000, gamma=1, alpha=0.1, epsilon='decay'):
        if epsilon == 'decay':
            def epsilon_fn(x): return 1 / (x + 1)
        else:
            def epsilon_fn(_): return epsilon

        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            done = False
            count = 0

            if np.random.random() < epsilon_fn(episode):
                action = self.env.action_space.sample()
            else:
                action = np.random.choice(
                    np.arange(16), p=softmax(self.Q[state]))

            while not done and count < self.TIMEOUT:
                count += 1

                next_state, reward, done, _ = self.env.step(action)

                if np.random.random() < epsilon_fn(episode):
                    next_action = self.env.action_space.sample()
                else:
                    next_action = np.random.choice(
                        np.arange(16), p=softmax(self.Q[next_state]))

                self.Q[state][action] = self.Q[state][action] + alpha * \
                    (reward + gamma *
                     self.Q[next_state][next_action] - self.Q[state][action])
                state = next_state
                action = next_action

    def actor_critic(self, episodes=1000, gamma=1, alpha=0.05, alpha_w=0.05):


        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            done = False
            count = 0

            while not done and count < self.TIMEOUT:
                action = np.random.choice(np.arange(16), p=softmax(self.Q[state]))
                count += 1
                next_state, reward, done, _ = self.env.step(action)
                delta = reward + gamma * (self.V[next_state])

        pass

    def play(self, print_grid=True, print_res=True):
        # Get initial state
        state, info = self.env.reset()
        done = False
        count = 0
        cumulative_reward = 0

        # Print the grid if needed
        if print_grid:
            print("Start", self.env.env._start)
            self.env.print_grid(self.env.grid)

        # For each timestep under the max limit
        while not done and count < self.play_limit:
            count += 1
            action = np.random.choice(np.arange(16), p=softmax(self.Q[state]))
            state, reward, done, _ = self.env.step(action)

            cumulative_reward += reward
            # Print additional info with the grids
            if print_grid:
                print("Action: ", action, "Reward: ", reward)
                self.env.print_grid(self.env.grid)

        # Print results
        if print_res:
            print("Start", self.env.env._start, "Count: ", count, "Reward: ", cumulative_reward)

        return count

    def play_many(self, world_count=10, sim_count=10):
        big_hist = []
        for _ in range(world_count):
            self.env.env.set_world()
            big_hist += [self.play(False, False) for _ in range(sim_count)]
        return np.array(big_hist)

    def set_rand_size(self, size):
        self.env.env.RAND_SIZE = size
        self.play_limit = size ** 2 * 3
        return size

    def get_rand_size(self):
        return self.env.env.RAND_SIZE

    def extract_policy(self):
        grid = self.env.grid
        policy_grid = np.zeros(grid.shape)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cur = [x,y]
                next_states = np.array([cur + action_to_direction[Action.WALK_UP], cur + action_to_direction[Action.WALK_LEFT],
                                        cur + action_to_direction[Action.WALK_DOWN], cur + action_to_direction[Action.WALK_RIGHT]])
                obj_contents = [GridTile.WALL.value if n[0] < 0 or n[1] < 0 or n[0] >=
                                grid.shape[0] or n[1] >= grid.shape[1] else grid[tuple(n)].value
                                for n in next_states]
                state = tuple(obj_contents)
                policy_grid[x][y] = np.where(self.Q[state] == np.max(self.Q[state]))[0][0]
        # print(policy_grid)
        self.env.print_grid(grid)
        print_policy(policy_grid)


def print_policy(grid):
    for x in range(grid.shape[0]):
        print("-------------------------")
        for y in range(grid.shape[1]):
            if grid[x][y] == Action.WALK_UP.value:
                print("| W/\ ", end="")
            if grid[x][y] == Action.WALK_LEFT.value:
                print("| W<  ", end="")
            if grid[x][y] == Action.WALK_DOWN.value:
                print("| W\/ ", end="")
            if grid[x][y] == Action.WALK_RIGHT.value:
                print("| W>  ", end="")

            if grid[x][y] == Action.OPEN_UP.value:
                print("| O/\ ", end="")
            if grid[x][y] == Action.OPEN_LEFT.value:
                print("| O<  ", end="")
            if grid[x][y] == Action.OPEN_DOWN.value:
                print("| O\/ ", end="")
            if grid[x][y] == Action.OPEN_RIGHT.value:
                print("| O>  ", end="")

            if grid[x][y] == Action.SWIM_UP.value:
                print("| S/\ ", end="")
            if grid[x][y] == Action.SWIM_LEFT.value:
                print("| S<  ", end="")
            if grid[x][y] == Action.SWIM_DOWN.value:
                print("| S\/ ", end="")
            if grid[x][y] == Action.SWIM_RIGHT.value:
                print("| S>  ", end="")

            if grid[x][y] == Action.JUMP_UP.value:
                print("| J/\ ", end="")
            if grid[x][y] == Action.JUMP_LEFT.value:
                print("| J<  ", end="")
            if grid[x][y] == Action.JUMP_DOWN.value:
                print("| J\/ ", end="")
            if grid[x][y] == Action.JUMP_RIGHT.value:
                print("| J>  ", end="")

        print("|")
    print("-------------------------")

def remove_outliers(data):
    medians = (data <= np.quantile(data, 0.8)) & (data >= np.quantile(data, 0.2))
    cleaned = data[np.where(medians)]
    return cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run')
    args = parser.parse_args()
    option = str(args.run)

    if option == "qlearn":
        a = Agent()

        a.extract_policy()
        # record of avg timesteps takene
        hist = []
        size = a.set_rand_size(4)
        for _ in range(50):
            print(f"\nEpisode {_}")
            a.q_learn()
            a.play(False, False)
            games = a.play_many(10)
            games_cleaned = remove_outliers(games)

            # mean timesteps taken
            mean_count = round(games_cleaned.mean(),2)

            print("Mean count: ", mean_count, "max", games_cleaned.max(), "limit", a.play_limit)

            if games_cleaned.max() != a.play_limit and mean_count < size**2/2:
                size = a.set_rand_size(a.get_rand_size() + 1)
                print("Increasing size, new size: ", size)
            a.env.env.set_world()
            hist.append(mean_count)

        a.play()
        a.extract_policy()

    if option == "montecarlo":
        a = Agent()
        # record of avg timesteps takene
        hist = []
        size = a.set_rand_size(4)
        for _ in range(50):
            print(f"\nEpisode {_}")
            a.monte_carlo()
            a.play(False, False)
            games = a.play_many(10)
            games_cleaned = remove_outliers(games)

            # mean timesteps taken
            mean_count = round(games_cleaned.mean(),2)

            print("Mean count: ", mean_count, "max",
                games_cleaned.max(), "limit", a.play_limit)
            if games_cleaned.max() != a.play_limit and mean_count < size**2/2:
                size = a.set_rand_size(a.get_rand_size() + 1)
                print("Increasing size, new size: ", size)
            a.env.env.set_world()
            hist.append(mean_count)

        a.play()

    if option == "sarsa":
        a = Agent()
        # record of avg timesteps takene
        hist = []
        size = a.set_rand_size(4)
        for _ in range(50):
            print(f"\nEpisode {_}")
            a.sarsa()
            a.play(False, False)
            games = a.play_many(10)
            games_cleaned = remove_outliers(games)

            # mean timesteps taken
            mean_count = round(games_cleaned.mean(),2)

            print("Mean count: ", mean_count, "max",
                games_cleaned.max(), "limit", a.play_limit)
            if games_cleaned.max() != a.play_limit and mean_count < size**2/2:
                size = a.set_rand_size(a.get_rand_size() + 1)
                print("Increasing size, new size: ", size)
            a.env.env.set_world()
            hist.append(mean_count)

        a.play()   
    if option == "actorcritic":
        a = Agent()
        # record of avg timesteps takene
        hist = []
        size = a.set_rand_size(4)
        for _ in range(50):
            print(f"\nEpisode {_}")
            a.actor_critic()
            a.play(False, False)
            games = a.play_many(10)
            games_cleaned = remove_outliers(games)

            # mean timesteps taken
            mean_count = round(games_cleaned.mean(),2)

            print("Mean count: ", mean_count, "max",
                games_cleaned.max(), "limit", a.play_limit)
            if games_cleaned.max() != a.play_limit and mean_count < size**2/2:
                size = a.set_rand_size(a.get_rand_size() + 1)
                print("Increasing size, new size: ", size)
            a.env.env.set_world()
            hist.append(mean_count)

        a.play()   
            
