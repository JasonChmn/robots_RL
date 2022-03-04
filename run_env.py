from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

import gym

#MODE_CONTROL = "PD"
MODE_CONTROL = "TORQUES"
if MODE_CONTROL=="PD":
    from Robots.envs.env_PD import Env_PD as Env
    ENV_ROBOT_ID = 'env_robot-v0' # Name in Robots/__init__.py
elif MODE_CONTROL=="TORQUES":
    from Robots.envs.env_torques import Env_torques as Env
    ENV_ROBOT_ID = 'env_robot-v1' # Name in Robots/__init__.py

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
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        env = [lambda: gym.make(ENV_ROBOT_ID)]
        print("env: ",env)
        envVec = DummyVecEnv(env)

    else:
        env = [make_env(ENV_ROBOT_ID, i) for i in range(n_procs)]
        envVec = SubprocVecEnv(env, start_method='spawn')
    print("=========== Create model")
    model = PPO2(MlpPolicy, envVec, verbose=1, tensorboard_log="./tensorboard/",
                 n_steps=2048, nminibatches=64, noptepochs=10, cliprange=0.2, learning_rate=1.0e-4,
                 gamma=0.98)
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
    env = Env(GUI=True)
    envVec = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, envVec)
    if path_model is not None:
        model.load(path_model, env=envVec, policy=MlpPolicy)
    obs = env.reset()
    counter, max_counter = 0, 200
    while True:
        action, _states  = model.predict(obs, deterministic=False)
        #print(counter," - action: ",[round(a,4) for a in action])
        #input("...")
        counter+=1
        obs, reward, done, _ = env.step( action )
        if done or counter>max_counter:
            #print("Max torques : ",env.robot.max_torques)
            input("Press to restart ...")
            env.reset()
            counter = 0
    pass


if __name__ == "__main__":
    # Basic tests
    #Env._run_test_env()
    #Env._run_test_solo()
    #Env._run_test_talos()
    # Other tests
    #Env._run_test_reset_solo()
    #Env._run_test_reset_talos()
    #Env._run_test_joints_solo()
    
    # Train or play
    TRAIN = False
    if TRAIN:
        train()
    else:
        model_name = None
        #model_name = "/devel/hpp/src/robots_RL/logs/solo_not_over"
        play(model_name)