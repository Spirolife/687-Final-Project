from gym.envs.registration import register

register(id="GridWorldEnv-v0",
         entry_point='sleep_environment.envs:GridWorldEnv')
