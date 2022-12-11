from tqdm import tqdm
import numpy as np
import gym
import sleep_environment
import CustomEnvironment.worlds as worlds
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

    def q_learn(self, episodes=10000, gamma=0.9, alpha=0.9, epsilon='decay'):
        self.Q = np.ones((7, 7, 7, 7, 16))
        if epsilon == 'decay':
            def epsilon_fn(x): return 1 / (x + 1)
        else:
            def epsilon_fn(_): return epsilon
        for episode in tqdm(range(episodes)):
            state, info = self.env.reset()
            done = False
            while not done:
                if np.random.random() < epsilon_fn(episode):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state])
                next_state, reward, done, _ = self.env.step(action)
                self.Q[state][action] = self.Q[state][action] + alpha * \
                    (reward + gamma *
                     np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state
        self.Q = self.Q

    def play(self):
        state, info = self.env.reset()
        done = False
        limit = 50
        count = 0
        print("Start", self.env.env._agent_location)
        self.env.print_grid(self.env.grid)
        while not done and count < limit:
            count += 1
            action = np.argmax(self.Q[state])
            state, reward, done, _ = self.env.step(action)

            print("Action: ", action, "Reward: ", reward)
            self.env.print_grid(self.env.grid)
        print("Done: ", done, "Count: ", count)


a = Agent(worlds.world6)
a.q_learn()
a.play()
a.play()
a.play()
