from gym.envs.registration import register

register(
     id='env_robot-v0',
     entry_point='Robots.envs:Env_PD',
     max_episode_steps=1500,
)

register(
     id='env_robot-v1',
     entry_point='Robots.envs:Env_torques',
     max_episode_steps=1500,
)