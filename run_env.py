from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

import gym

from Robots.envs.env0 import Env0

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

def train():
    n_procs = 8
    env_id = 'env_robot-v0'
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        env = [lambda: gym.make(env_id)]
        print("env: ",env)
        envVec = DummyVecEnv(env)

    else:
        env = [make_env(env_id, i) for i in range(n_procs)]
        envVec = SubprocVecEnv(env, start_method='spawn')
    print("=========== Create model")
    model = PPO2(MlpPolicy, envVec, verbose=1, tensorboard_log="./tensorboard/",
                 n_steps=1024, nminibatches=64, noptepochs=10, cliprange=0.2, learning_rate=1.0e-4,
                 gamma=0.99)
    print("=========== Create callback")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/',
                                             name_prefix="model_")
    print("=========== Learn")
    model.learn(total_timesteps=int(1e8), callback=checkpoint_callback)
    input("Training over...")
    #del model
    env.close()
    pass

def play(path_model=None):
    env = Env0(GUI=True)
    envVec = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, envVec, verbose=1, tensorboard_log="./tensorboard/",
                 n_steps=2048, nminibatches=64, noptepochs=10, cliprange=0.2, learning_rate=1.0e-4,
                 gamma=0.99)
    if path_model is not None:
        model.load(path_model, env=envVec, policy=MlpPolicy)
    obs = env.reset()
    counter, max_counter = 0, 200
    while True:
        action, _states  = model.predict(obs, deterministic=False)
        print(counter," - action: ",action)
        counter+=1
        obs, reward, done, _ = env.step( action )
        if done or counter>max_counter:
            input("Press to restart ...")
            env.reset()
            counter = 0
    pass


if __name__ == "__main__":
    #Env0._run_test_solo()
    #Env0._run_test_talos()
    TRAIN = False
    if TRAIN:
        train()
    else:
        #model_name = "/devel/hpp/src/robots_RL/logs/model__2000000_steps"
        play(model_name)