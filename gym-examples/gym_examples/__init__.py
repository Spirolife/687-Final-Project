from gymnasium.envs.registration import register
import gymnasium as gym

register(
    id='CustomEnv-v0',
    entry_point='GridWorldEnv',
    max_episode_steps=1000,
)
