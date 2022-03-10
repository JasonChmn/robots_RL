from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy as CustomPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import gym

from Robots.ressources.config import Config
if Config.MODE_CONTROL=="PD":
    from Robots.envs.env_PD import Env_PD as Env
elif Config.MODE_CONTROL=="TORQUE":
    from Robots.envs.env_torque import Env_torque as Env
else:
    input("ERROR, NO CONTROL SELECTED IN CONFIG...")


# For multiprocessing
from stable_baselines.common import set_global_seeds
def make_env(env_id, rank, seed=0):
    #Utility function for multiprocessed env.
    #:param env_id: (str) the environment ID
    #:param seed: (int) the inital seed for RNG
    #:param rank: (int) index of the subprocess
    def _init():
        env = gym.make(env_id)
        # Important: use a different seed for each environment
        print("===========================================> SETTING SEED : ",seed+rank)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

"""
from stable_baselines.common.policies import FeedForwardPolicy
class CustomPolicy(FeedForwardPolicy):
    name = 'CustomPolicy'
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128,128,128],
                                                          vf=[128,128,128]
                                                         )
                                                    ],
                                           feature_extraction="mlp"
                                          )
"""


def getModel(envVec):
    if Config.MODE_CONTROL=="PD": gamma = 0.97 # PD
    else:                         gamma = 0.97 # Torques
    model = PPO2(CustomPolicy, envVec, verbose=1, tensorboard_log="./tensorboard/",
                 n_steps=2048, nminibatches=256, noptepochs=10, cliprange=0.2, learning_rate=1.0e-4,
                 gamma=gamma)
    return model

# ===========================================================================================

def train():
    n_procs = 8
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        env = [lambda: gym.make(Config.ENV_ROBOT_ID)]
        print("env: ",env)
        envVec = DummyVecEnv(env)
    else:
        env = [make_env(Config.ENV_ROBOT_ID, i) for i in range(n_procs)]
        envVec = SubprocVecEnv(env, start_method='spawn')
    print("=========== Create model")
    model = getModel(envVec)
    print("=========== Create callback")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                             name_prefix="model_")
    print("=========== Learn")
    model.learn(total_timesteps=int(1e8), callback=checkpoint_callback)
    input("Training over...")
    #del model
    env.close()
    return None


def play(path_model=None):
    env = Env(GUI=True)
    envVec = DummyVecEnv([lambda: env])
    if path_model is not None:
        print("Load model : ",path_model)
        model = PPO2.load(path_model, env=envVec, policy=CustomPolicy)
    else:
        print("Null model")
        model = getModel(envVec)
    obs = env.reset()
    counter, max_counter = 0, 200
    while True:
        action, _states  = model.predict(obs, deterministic=True)
        #print(counter," - action: ",[round(a,4) for a in action])
        #input("...")
        counter+=1
        obs, reward, done, _ = env.step( action )
        if done or counter>max_counter:
            #print("Max torques : ",env.robot.max_torques)
            input("Press to restart ...")
            env.reset()
            counter = 0
    return None


# ===========================================================================================

if __name__ == "__main__":
    # TEST - environment, move robot with null action
    #Env._run_test_env()
    # TEST - on robot class
    #Env._run_test_robot()        # Move to default position.
    #Env._run_test_robot_reset()  # Check if the robot is reset correctly after each episode.
    #Env._run_test_robot_joints() # Check bounds for each controllable joint of the robot
    
    # Train or play
    TRAIN = False
    if TRAIN:
        train()
    else:
        #model_name = None
        #model_name = "/devel/hpp/src/robots_RL/logs/model__880000_steps"
        model_name = "/devel/hpp/src/robots_RL/logs_solo_stand_torque/model__880000_steps" 
        play(model_name)