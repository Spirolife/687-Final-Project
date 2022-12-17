import argparse
from tqdm import tqdm
import numpy as np
import gym
import sleep_environment
import CustomEnvironment.worlds as worlds
from enums import GridTile, Action, Observation, action_to_direction
from scipy.special import softmax
from matplotlib import pyplot as plt
import gym_toytext
from queue import PriorityQueue
from collections import defaultdict
import time

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
    def __init__(self, problem, world=None):
        if problem == "grid":
            self.env = gym.make("GridWorldEnv-v0", world=world)
            size = [x.n for x in self.env.observation_space.values()]
            size.append(self.env.action_space.n)
            size = tuple(size)
            self.Q = np.zeros(size)
            self.P = np.zeros([x.n for x in self.env.observation_space.values()])
        else:
            self.env = gym.make("Roulette-v0")
            self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            self.P = np.zeros(self.env.observation_space.n)

        self.problem = problem
        self.set_rand_size(4)
    # For Actor-Critic
        # self.AC_P = np.zeros(())

    TIMEOUT = 1000

    def q_learn(self, episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.3):
        if epsilon == 'decay':
            def epsilon_fn(x): return 1 / (x + 1)
        else:
            def epsilon_fn(_): return epsilon
        
        timesteps = []
        rewards = []

        for episode in tqdm(range(episodes)):
            if self.problem == "roulette":
                state = self.env.reset()
            else:
                state, info = self.env.reset()
            done = False
            count = 0
            cur_reward = 0
            while not done and count < self.TIMEOUT:
                count += 1
                if np.random.random() < epsilon_fn(episode):
                    action = self.env.action_space.sample()
                else:
                    action = np.random.choice(
                        np.arange(self.env.action_space.n), p=softmax(self.Q[state]))

                next_state, reward, done, _ = self.env.step(action)
                cur_reward += reward

                self.Q[state][action] = self.Q[state][action] + alpha * \
                    (reward + gamma *
                     np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state

            rewards.append(cur_reward)
            timesteps.append(count)

        return timesteps, rewards
            

    def monte_carlo(self, episodes=10000, epsilon='decay'):
        returns = {}
        timesteps = []
        rewards = []
        for episode in tqdm(range(episodes)):
            # timesteps x [state action reward]
            seen_sa = []
            trajectory = []
            if problem == 'roulette':
                state = self.env.reset()
            else:
                state, info = self.env.reset()
            # print(state)
            done = False
            count = 0
            cur_reward = 0

            while not done and count < self.TIMEOUT:
                if count == 0:
                    action = self.env.action_space.sample()
                else:
                    action = np.where(self.Q[state] == np.max(self.Q[state]))[0][0]
    
                count += 1

                next_state, reward, done, _ = self.env.step(action)
                cur_reward += reward
                trajectory.append([state, action, reward])

                state = next_state

            rewards.append(cur_reward)
            timesteps.append(count)
            # print(returns)

            returns_cumulative = np.flip(np.cumsum(np.flip(np.asarray(trajectory)[:,2])))

            for t in range(len(trajectory)-1):
                # print(t)
                state = trajectory[t][0]
                action = trajectory[t][1]
                # If this state action pair has not been seen
                if (state, action) not in seen_sa:
                    # add it to seen
                    seen_sa.append((state, action))
                    
                    # Enter the return into the dict
                    if (state, action) in returns:
                        returns[state,action].append(returns_cumulative[t])
                    else:
                        returns[state,action] = [returns_cumulative[t]]

                    # Update the q function
                    self.Q[state][action] = np.average(returns[state, action])

        return timesteps, rewards

    def sarsa(self, episodes=1000, gamma=1, alpha=0.1, epsilon='decay'):
        if epsilon == 'decay':
            def epsilon_fn(x): return 1 / (x + 1)
        else:
            def epsilon_fn(_): return epsilon

        timesteps = []
        rewards = []

        for episode in tqdm(range(episodes)):
            if problem == 'roulette':
                state = self.env.reset()
            else:
                state, info = self.env.reset()

            done = False
            count = 0
            cur_reward = 0

            if np.random.random() < epsilon_fn(episode):
                action = self.env.action_space.sample()
            else:
                action = np.random.choice(
                    np.arange(self.env.action_space.n), p=softmax(self.Q[state] * 100))

            while not done and count < self.TIMEOUT:
                count += 1
                next_state, reward, done, _ = self.env.step(action)
                cur_reward += reward

                if np.random.random() < epsilon_fn(episode):
                    next_action = self.env.action_space.sample()
                else:
                    next_action = np.random.choice(
                        np.arange(self.env.action_space.n), p=softmax(self.Q[next_state] * 100))


                if done:
                    self.Q[state][action] = self.Q[state][action] + alpha * \
                    (reward - self.Q[state][action])
                else:
                    self.Q[state][action] = self.Q[state][action] + alpha * \
                        (reward + gamma * self.Q[next_state][next_action] - self.Q[state][action])

                state = next_state
                action = next_action

            rewards.append(cur_reward)
            timesteps.append(count)

        return timesteps, rewards

    def sarsalambda(self, episodes=1000, gamma=0.9, alpha=0.1, lambda_decay=0.9):
        e = np.zeros(self.Q.shape)
        if self.problem == 'roulette':
            states_list = np.arange(self.env.observation_space.n)
        else:
            states_list = self.get_all_states(self.env.grid)
        timesteps = []
        rewards = []

        for episode in tqdm(range(episodes)):
            if problem == 'roulette':
                state = self.env.reset()
            else:
                state, info = self.env.reset()            
            done = False
            count = 0
            cur_reward = 0
            action = np.random.choice(np.arange(self.env.action_space.n), p=softmax(self.Q[state] * 100))

            while not done and count < self.TIMEOUT:
                count += 1
                next_state, reward, done, _ = self.env.step(action)
                cur_reward += reward
                next_action = np.random.choice(np.arange(self.env.action_space.n), p=softmax(self.Q[next_state] * 100))

                delta = reward + gamma * self.Q[next_state][next_action] - self.Q[state][action]
                e[state][action] = e[state][action] + 1

                for state in states_list:
                    for a in range(self.env.action_space.n):
                        self.Q[state][a] = self.Q[state][a] + alpha * delta * e[state][a]
                        e[state][a] = gamma * lambda_decay * e[state][a]

                state = next_state
                action = next_action
            rewards.append(cur_reward)
            timesteps.append(count)

        return timesteps, rewards

    def prioritize_sweep(self, episodes=1000, gamma=1, alpha=0.1, epsilon='decay', theta=2):

            model = {}
            model_back = defaultdict(set)
            PQueue = PriorityQueue()

            if epsilon == 'decay':
                def epsilon_fn(x): return 1 / (x + 1)
            else:
                def epsilon_fn(_): return epsilon

            timesteps = []
            rewards = []

            for episode in tqdm(range(episodes)):
                reward_sum = 0
                if problem == 'roulette':
                    cur_state = self.env.reset()
                else:
                    cur_state, info = self.env.reset()

                done = False
                count = 0
                while not done and count < self.TIMEOUT:
                    count += 1
                    if np.random.random() < epsilon_fn(episode):
                        action = self.env.action_space.sample()
                    else:
                        action = np.random.choice(
                            np.arange(self.env.action_space.n), p=softmax(self.Q[cur_state]))

                    next_state, reward, done, _ = self.env.step(action)
                    reward_sum += reward

                    model[(cur_state, action)] = (next_state, reward, done)
                    model_back[next_state].add((cur_state, action, reward))

                    priority = abs(
                        reward + gamma * np.max(self.Q[next_state]) - self.Q[cur_state][action])

                    if priority > theta:
                        PQueue.put((priority, (cur_state, action)))

                    limit = 5
                    while not PQueue.empty() and limit > 0:
                        limit -= 1
                        _, (state, action) = PQueue.get()
                        next_state, reward, _ = model[(state, action)]
                        self.Q[state][action] = self.Q[state][action] + alpha * \
                            (reward + gamma *
                            np.max(self.Q[next_state]) - self.Q[state][action])
                        for prev_state, prev_action, r in model_back[state]:
                            priority = abs(
                                r + gamma * np.max(self.Q[state]) - self.Q[prev_state][prev_action])
                            if priority > theta:
                                PQueue.put((priority, (prev_state, prev_action)))

                timesteps.append(count)
                rewards.append(reward_sum)
            return timesteps, rewards

    def play(self, print_grid=True, print_res=True):
        # Get initial state
        if self.problem == 'roulette':
            state = self.env.reset()
        else:
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
            action = np.random.choice(np.arange(self.env.action_space.n), p=softmax(self.Q[state] * 1000))
            state, reward, done, _ = self.env.step(action)

            cumulative_reward += reward
            # Print additional info with the grids
            if print_grid:
                print("Action: ", action, "Reward: ", reward)
                self.env.print_grid(self.env.grid)

        # Print results
        if print_res:
            if problem == 'roulette':
                print("Count: ", count, "Reward: ", cumulative_reward)
            else:
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
        self.play_limit = 100
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
                if grid[x][y].value == GridTile.WALL.value:
                    policy_grid[x][y] = -1
                else:
                    policy_grid[x][y] = np.where(self.Q[state] == np.max(self.Q[state]))[0][0]
        print(grid)
        print(policy_grid)
        self.env.print_grid(grid)
        print_policy(policy_grid)

    def get_all_states(self, grid):
        states_list = []
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cur = [x,y]
                if grid[x,y] == GridTile.WALL:
                    continue
                next_states = np.array([cur + action_to_direction[Action.WALK_UP], cur + action_to_direction[Action.WALK_LEFT],
                                        cur + action_to_direction[Action.WALK_DOWN], cur + action_to_direction[Action.WALK_RIGHT]])
                obj_contents = [GridTile.WALL.value if n[0] < 0 or n[1] < 0 or n[0] >=
                                grid.shape[0] or n[1] >= grid.shape[1] else grid[tuple(n)].value
                                for n in next_states]
                state = tuple(obj_contents)
                states_list.append(state)
        return states_list

def print_policy(grid):
    for x in range(grid.shape[0]):
        print("-------------------------------")
        for y in range(grid.shape[1]):
            if grid[x][y] == -1:
                print("||||||", end="")

            if grid[x][y] == Action.WALK_UP.value:
                print("| W/\ ", end="")
            elif grid[x][y] == Action.WALK_LEFT.value:
                print("| W<  ", end="")
            elif grid[x][y] == Action.WALK_DOWN.value:
                print("| W\/ ", end="")
            elif grid[x][y] == Action.WALK_RIGHT.value:
                print("| W>  ", end="")

            elif grid[x][y] == Action.OPEN_UP.value:
                print("| O/\ ", end="")
            elif grid[x][y] == Action.OPEN_LEFT.value:
                print("| O<  ", end="")
            elif grid[x][y] == Action.OPEN_DOWN.value:
                print("| O\/ ", end="")
            elif grid[x][y] == Action.OPEN_RIGHT.value:
                print("| O>  ", end="")

            elif grid[x][y] == Action.SWIM_UP.value:
                print("| S/\ ", end="")
            elif grid[x][y] == Action.SWIM_LEFT.value:
                print("| S<  ", end="")
            elif grid[x][y] == Action.SWIM_DOWN.value:
                print("| S\/ ", end="")
            elif grid[x][y] == Action.SWIM_RIGHT.value:
                print("| S>  ", end="")

            elif grid[x][y] == Action.JUMP_UP.value:
                print("| J/\ ", end="")
            elif grid[x][y] == Action.JUMP_LEFT.value:
                print("| J<  ", end="")
            elif grid[x][y] == Action.JUMP_DOWN.value:
                print("| J\/ ", end="")
            elif grid[x][y] == Action.JUMP_RIGHT.value:
                print("| J>  ", end="")

        print("|")
    print("-------------------------------")

def remove_outliers(data):
    medians = (data <= np.quantile(data, 0.8)) & (data >= np.quantile(data, 0.2))
    cleaned = data[np.where(medians)]
    return cleaned

def print_learning_curve(tntcomplete):
    x = np.arange(len(tntcomplete))
    y = tntcomplete
    plt.title("Plot")
    plt.xlabel("episode")
    plt.ylabel("number of timesteps needed to complete")
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run')
    parser.add_argument('-p', '--problem')
    args = parser.parse_args()
    option = str(args.run)
    problem = str(args.problem)

    a = Agent(problem)

    if problem == "grid":
        # record of avg timesteps takene
        hist = []
        num_hundreds_hist = []
        rewards = ""

        size = a.set_rand_size(5)
        for _ in range(100):
            print(f"\nEpisode {_}")
            if option == "Q Learning":
                tntcomplete, rewards = a.q_learn()
            elif option == 'First Visit Monte Carlo':
                tntcomplete, rewards = a.monte_carlo()
            elif option == 'SARSA':
                tntcomplete, rewards = a.sarsa()
            elif option == 'SARSA(L)':
                tntcomplete, rewards = a.sarsalambda()
            elif option == 'Prioritize Sweep':
                tntcomplete, rewards = a.prioritize_sweep()
            else:
                print("Error: no algorithm selected.")
                exit()

            a.extract_policy()
            
            # print_learning_curve(tntcomplete)
            a.play(False, False)

            games_cleaned = a.play_many()
                
            # games_cleaned = remove_outliers(games)

            # mean timesteps taken
            mean_count = round(games_cleaned.mean(), 2)
            num_hundreds = np.count_nonzero(np.asarray(games_cleaned) == 100)

            print("Mean count: ", mean_count, "max", games_cleaned.max(), "limit", a.play_limit, "num_hundreds: ", num_hundreds)
            a.env.env.set_world()
            hist.append(mean_count)
            num_hundreds_hist.append(num_hundreds)

        plt.title("Timesteps Until Completion for " + option)
        plt.xlabel("Episode")
        plt.ylabel("Time Needed to Complete")
        plt.plot(tntcomplete)
        plt.show()


        plt.title("Rewards for " + option)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(rewards)
        plt.show()

        plt.title("Average Completion Time for Arbitrary Worlds")
        plt.xlabel("Round")
        plt.ylabel("Avg Completion Time")
        plt.plot(np.arange(len(hist)), hist)
        plt.show()

        plt.title("Number of 100 Occurrences for Arbitrary Worlds")
        plt.xlabel("Round")
        plt.ylabel("Occurrences")
        plt.plot(num_hundreds_hist)
        plt.show()

        a.play()
        a.extract_policy()

    if problem == 'roulette':
        tntcomplete, rewards = a.prioritize_sweep(episodes=100)

        # a.play(False, True)
        
        plt.title("Timesteps Until Completion for Roulette")
        plt.xlabel("Episode")
        plt.ylabel("Time Needed to Complete")
        plt.plot(tntcomplete)
        plt.show()

        pass
        # plt.title("Rewards for " + option)
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.plot(rewards)
        # plt.show()
