from gym.envs.registration import register
from Robots.ressources.config import Config

register(
     id='env_robot-v0',
     entry_point='Robots.envs:Env_PD',
     max_episode_steps=Config.NB_MAX_STEPS,
)

register(
     id='env_robot-v1',
     entry_point='Robots.envs:Env_torque',
     max_episode_steps=Config.NB_MAX_STEPS,
)