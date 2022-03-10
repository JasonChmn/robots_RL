import gym
import numpy as np

from Robots.ressources.plane import Plane # Terrain selected
from Robots.ressources.config import Config

if Config.NAME_ROBOT=="talos":
    from Robots.ressources.talos import Talos as Robot
elif Config.NAME_ROBOT=="solo":
    from Robots.ressources.solo  import Solo  as Robot
else:
    input("ERROR, NO ROBOT SELECTED IN CONFIG...")

# ==================================================================================

class Env_PD(gym.Env):
    metadata = {'render.modes': ['_']} # Not used

    def __init__(self, GUI=False):
        self.REAL_TIME = True and GUI
        self.robot = Robot(class_terrain=Plane, GUI=GUI)
        # State and Action space
        # Read : https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
        #        section : Tips and Tricks when creating a custom environment
        # Observation => All joints(q_mes+v_mes) + Base(base_pos+base_ori+base_lin_vel+base_ang_vel)
        q_mes, v_mes = self.robot.getJointsState()
        base_pos, base_ori = self.robot.getBasePosOri()
        base_lin_vel, base_ang_vel = self.robot.getBaseVel()
        self._len_q_mes, self._len_v_mes       = len(q_mes), len(v_mes)
        self._len_base_pos, self._len_base_ori = len(base_pos), len(base_ori)
        if Config.IS_POS_BASE_ONE_DIMENSION: self._len_base_pos=1
        self._len_base_lin_vel, self._len_base_ang_vel = len(base_lin_vel), len(base_ang_vel)
        self.obs_dim  = self._len_q_mes+self._len_v_mes+self._len_base_pos                  # number of joints * 2 + _len_base_pos
        self.obs_dim += self._len_base_ori+self._len_base_lin_vel+self._len_base_ang_vel    # 11 or 13, depending on IS_POS_BASE_ONE_DIMENSION
        self.observation_space = gym.spaces.box.Box(
            low=-1,
            high=1,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        # Action => All controlled joints
        # Two actions possible : q_desired and v_desired
        # We have two modes : One with and one without v_desired. If without, we fix v_desired to 0
        self._len_controlled_joints_state = len(self.robot.getControlledJointsState()[0])
        if Config.Q_DESIRED_ONLY:
            self.action_dim = self._len_controlled_joints_state     # q_desired
        else:
            self.action_dim = self._len_controlled_joints_state*2   # q_desired + v_desired
        self.action_space = gym.spaces.box.Box(
            low=-1,
            high=1,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        """
        # Check if these values are correct for action_dim and obs_dim
        print("nb_controlled_joints : ",len(self.robot.getControlledJointsState()[0]))
        print("all_joints : ",len(self.robot.getJointsState()[0]))
        if IS_POS_BASE_ONE_DIMENSION:
            print("obs_dim should be    : all_joints*2 + 11     => ",self.obs_dim)
        else:
            print("obs_dim should be    : all_joints*2 + 13     => ",self.obs_dim)
        print("action_dim should be : nb_controlled_joints  => ",self.action_dim)
        """
        # Bounds of robot base
        self.bound_base_pos = [ [-20,20], [-20,20], [-0.2, Config.HEIGHT_ROOT+0.3] ] # TO MODIFY
        self.bound_base_ori = [[-1,1]]*4    # Quaternion
        self.bound_base_lin_vel = [ [-1,1], [-1,1], [-1,1] ]                  # TO MODIFY
        self.bound_base_ang_vel = [[-4*np.pi,4*np.pi]]*3                      # TO MODIFY
        # Reset
        self.reset()
        pass


    def reset(self):
        self.robot.reset()
        obs, obs_normalized = self.getObservation()
        return np.array(obs_normalized)


    def step(self, action):
        # Unnormalize action
        action_unnormalized = self.unnormalizeAction(action.tolist())
        if Config.Q_DESIRED_ONLY:
            q_des = action_unnormalized
            v_des = [0.]*self._len_controlled_joints_state
        else:
            q_des = action_unnormalized[0:self._len_controlled_joints_state]
            v_des = action_unnormalized[self._len_controlled_joints_state:-1]
        # Move the robot
        self.robot.moveRobot(np.array(q_des),np.array(v_des), real_time=self.REAL_TIME)
        # Get updated state
        obs, obs_normalized = self.getObservation()
        # Get reward
        reward = self.getReward()
        # Check terminal condition
        done = self.checkDoneCondition()
        return np.array(obs_normalized), reward, done, {}

    # ======================================================================================

    # This function returns an observation and a normalized observation of the robot.
    def getObservation(self):
        obs, obs_normalized = [], []
        # Observation => All joints(q_mes+v_mes) + Base(base_pos+base_ori+base_lin_vel+base_ang_vel)
        q_mes, v_mes = self.robot.getJointsState()
        base_pos, base_ori = self.robot.getBasePosOri()
        if Config.IS_POS_BASE_ONE_DIMENSION: base_pos = base_pos[2]
        base_lin_vel, base_ang_vel = self.robot.getBaseVel()
        # q_mes
        q_mes_normalized = q_mes.copy()
        for i in range(0,self._len_q_mes):
            q_mes_normalized[i] = Env_PD._rescale(q_mes_normalized[i], self.robot.joints_bound_pos_all[i], [-1,1])
        obs += q_mes
        obs_normalized += q_mes_normalized
        # v_mes
        v_mes_normalized = v_mes.copy()
        for i in range(0,self._len_v_mes):
            v_mes_normalized[i] = Env_PD._rescale(v_mes_normalized[i], self.robot.joints_bound_vel_all[i], [-1,1])
        obs += v_mes
        obs_normalized += v_mes_normalized
        # base_pos
        if Config.IS_POS_BASE_ONE_DIMENSION:
            base_pos_normalized = Env_PD._rescale(base_pos,self.bound_base_pos[2],[-1,1])
            obs += [base_pos]
            obs_normalized += [base_pos_normalized]
        else:
            base_pos_normalized = base_pos.copy()
            for j in range(0,self._len_base_pos):
                base_pos_normalized[i] = Env_PD._rescale(base_pos_normalized[i],self.bound_base_pos[i],[-1,1])
            obs += base_pos
            obs_normalized += base_pos_normalized
        # base_ori
        base_ori_normalized = base_ori.copy()
        for i in range(0,self._len_base_ori):
            base_ori_normalized[i] = Env_PD._rescale(base_ori_normalized[i],self.bound_base_ori[i],[-1,1])
        obs += base_ori
        obs_normalized += base_ori_normalized
        # base_lin_vel
        base_lin_vel_normalized = base_lin_vel.copy()
        for i in range(0,self._len_base_lin_vel):
            base_lin_vel_normalized[i] = Env_PD._rescale(base_lin_vel_normalized[i],self.bound_base_lin_vel[i],[-1,1])
        obs += base_lin_vel
        obs_normalized += base_lin_vel_normalized
        # base_ang_vel
        base_ang_vel_normalized = base_ang_vel.copy()
        for i in range(0,self._len_base_ang_vel):
            base_ang_vel_normalized[i] = Env_PD._rescale(base_ang_vel_normalized[i],self.bound_base_ang_vel[i],[-1,1])
        obs += base_ang_vel
        obs_normalized += base_ang_vel_normalized
        return obs, obs_normalized

    def getReward(self, print_info=False):
        reward = 0.
        # Keep the robot standing (fixed base position on Z)
        base_pos, _ = self.robot.getBasePosOri()
        distance_height_to_treshold = max(Config.HEIGHT_ROOT-Config.TRESHOLD_DEAD[0], Config.TRESHOLD_DEAD[1]-Config.HEIGHT_ROOT)
        R_pos = 1.0 - min(distance_height_to_treshold, abs(Config.HEIGHT_ROOT-base_pos[2]))/distance_height_to_treshold # Positive reward
        reward = R_pos
        if print_info:
            print("REWARDS (without weights) : pos=",reward(R_pos,2))
        return reward / Config.NB_MAX_STEPS

    def checkDoneCondition(self):
        done = False
        # Keep the robot root above a treshold
        base_pos, _ = self.robot.getBasePosOri()
        if base_pos[2]<Config.TRESHOLD_DEAD[0] or base_pos[2]>Config.TRESHOLD_DEAD[1]:
            done=True
            print("Episode done, threshold: ",Config.TRESHOLD_DEAD," and position z: ",base_pos[2])
        return done

    # ======================================================================================

    # This needs to be optimized in the future.
    def unnormalizeAction(self, action_normalized):
        action = action_normalized.copy()
        number_controlled_joints = len(self.robot.controlled_joints)
        # unnormalize q_des
        for i in range(0,number_controlled_joints):
            action[i] = Env_PD._rescale( action[i], [-1,1], self.robot.joints_bound_pos[i] )
        # if activated, unnormalize v_des
        if not Config.Q_DESIRED_ONLY:
            for i in range(0,number_controlled_joints):
                action[i+number_controlled_joints] = Env_PD._rescale( action[i+number_controlled_joints], [-1,1], self.robot.joints_bound_pos[i] )
        return action

    # ======================================= TOOLS

    # _rescale a value in input_bounds to output_bounds.
    #   @input  value           : value to _rescale
    #           input_bounds    : range of value
    #           output_bounds   : range desired
    #   @output value _rescaled in output_bounds
    @staticmethod
    def _rescale(value, input_bounds, output_bounds):
        delta1 = input_bounds[1] - input_bounds[0]
        delta2 = output_bounds[1] - output_bounds[0]
        if delta1==0:
            return output_bounds[0]+delta2/2.
        else:
            return (delta2 * (value - input_bounds[0]) / delta1) + output_bounds[0]

    # ======================================= CLASSES TEST

    # Test the environments : Actions are all [0.]
    @staticmethod
    def _run_test_env():
        env = Env_PD(GUI=True)
        action = np.array([0.]*env.action_dim)
        i, counter = 0, 100
        while True:
            obs, reward, done, _ = env.step( action )
            i+=1
            print("step ",i," / ",counter)
            if i%counter==0:
                done = True
                print("Reach max number of configs : ",counter)
            if done:
                input("Press to restart ...")
                env.reset()
                i = 0
        return None
    
    # Test Robot class : Move to default position
    @staticmethod
    def _run_test_robot():
        Robot._run_test()
        return None

    # ======================================= OTHER TESTS

    # Test Robot class : Check if the robot is reset correctly after each episode.
    @staticmethod
    def _run_test_robot_reset():
        Robot._run_test_reset()
        return None

    # Test Robot class : Check bounds for each controllable joint of the robot
    @staticmethod
    def _run_test_robot_joints():
        Robot._run_test_joints_limit()
        return None
