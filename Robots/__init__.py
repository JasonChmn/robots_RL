from gym.envs.registration import register

register(
     id='env_robot-v0',
     entry_point='Robots.envs:Env0',
     max_episode_steps=1500,
)