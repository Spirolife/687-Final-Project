from tqdm import tqdm
import numpy as np
import gym
import sleep_environment
import CustomEnvironment.worlds as worlds
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
        self.Q = np.ones((7, 7, 7, 7, 16))
        self.set_rand_size(4)

    TIMEOUT = 1000

    def q_learn(self, episodes=1000, gamma=1, alpha=0.1, epsilon='decay'):
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
        self.Q = self.Q

    def play(self, print_grid=True, print_res=True):
        state, info = self.env.reset()
        done = False
        count = 0
        if print_grid:
            print("Start", self.env.env._start)
            self.env.print_grid(self.env.grid)
        while not done and count < self.play_limit:
            count += 1
            action = np.random.choice(np.arange(16), p=softmax(self.Q[state]))
            state, reward, done, _ = self.env.step(action)

            if print_grid:
                print("Action: ", action, "Reward: ", reward)
                self.env.print_grid(self.env.grid)
        if print_res:
            print("Start", self.env.env._start, "Count: ", count)
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


def remove_outliers(data):
    medians = (data <= np.quantile(data, 0.8)) & (data >= np.quantile(data, 0.2))
    cleaned = data[np.where(medians)]
    return cleaned


a = Agent()
hist = []
size = a.set_rand_size(4)
for _ in range(500):
    print(f"\nEpisode {_}")
    a.q_learn()
    a.play(False, False)
    games = a.play_many(10)
    games_cleaned = remove_outliers(games)
    mean_count = round(games_cleaned.mean(),2)
    print("Mean count: ", mean_count, "max",
          games_cleaned.max(), "limit", a.play_limit)
    if games_cleaned.max() != a.play_limit and mean_count < size**2/2:
        size = a.set_rand_size(a.get_rand_size() + 1)
        print("Increasing size, new size: ", size)
    a.env.env.set_world()
    hist.append(mean_count)

a.play()
a.play()
a.play()
