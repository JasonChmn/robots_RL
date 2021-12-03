import gym
import numpy as np

from Robots.ressources.plane import Plane # Terrain selected

ROBOT_LIST = ["talos","solo"]
#NAME_ROBOT = ROBOT_LIST[0] # talos
NAME_ROBOT = ROBOT_LIST[1] # solo
if NAME_ROBOT=="talos":
    from Robots.ressources.talos import Talos as Robot
    HEIGHT_ROOT = 1.0   # Height of the robot when standing
    TRESHOLD_DEAD = [HEIGHT_ROOT-0.3, 2.0]  # Episode is over if the robot goes lower or higher
    DIVIDE_ACTION_POS_VALUE = 1.
elif NAME_ROBOT=="solo":
    from Robots.ressources.solo import Solo   as Robot
    HEIGHT_ROOT = 0.23  # Height of the robot when standing
    TRESHOLD_DEAD = [HEIGHT_ROOT-0.08, 0.6] # Episode is over if the robot goes lower or higher
    DIVIDE_ACTION_POS_VALUE = 2.0 # In URDF, it's [-10,10] radians
else:
    input("Error, name of the robot not defined ...")

Q_DESIRED_ONLY = True
IS_POS_BASE_ONE_DIMENSION = True # If true, we keep only the Z value of base position

# ==================================================================================


class Env0(gym.Env):
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
        if IS_POS_BASE_ONE_DIMENSION: self._len_base_pos=1
        self._len_base_lin_vel, self._len_base_ang_vel = len(base_lin_vel), len(base_ang_vel)
        self.obs_dim  = self._len_q_mes+self._len_v_mes+self._len_base_pos                  # number of joints * 2
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
        if Q_DESIRED_ONLY:
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
        self.bound_base_pos = [ [-20,20], [-20,20], [-0.2, HEIGHT_ROOT+0.3] ] # TO MODIFY
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
        if Q_DESIRED_ONLY:
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
        if IS_POS_BASE_ONE_DIMENSION: base_pos = base_pos[2]
        base_lin_vel, base_ang_vel = self.robot.getBaseVel()
        # q_mes
        q_mes_normalized = q_mes[::]
        for i in range(0,self._len_q_mes):
            q_mes_normalized[i] = Env0._rescale(q_mes_normalized[i], self.robot.joints_bound_pos_all[i], [-1,1])
        obs += q_mes
        obs_normalized += q_mes_normalized
        # v_mes
        v_mes_normalized = v_mes[::]
        for i in range(0,self._len_v_mes):
            v_mes_normalized[i] = Env0._rescale(v_mes_normalized[i], self.robot.joints_bound_vel_all[i], [-1,1])
        obs += v_mes
        obs_normalized += v_mes_normalized
        # base_pos
        if IS_POS_BASE_ONE_DIMENSION:
            base_pos_normalized = Env0._rescale(base_pos,self.bound_base_pos[2],[-1,1])
            obs += [base_pos]
            obs_normalized += [base_pos_normalized]
        else:
            base_pos_normalized = base_pos[::]
            for j in range(0,self._len_base_pos):
                base_pos_normalized[i] = Env0._rescale(base_pos_normalized[i],self.bound_base_pos[i],[-1,1])
            obs += base_pos
            obs_normalized += base_pos_normalized
        # base_ori
        base_ori_normalized = base_ori[::]
        for i in range(0,self._len_base_ori):
            base_ori_normalized[i] = Env0._rescale(base_ori_normalized[i],self.bound_base_ori[i],[-1,1])
        obs += base_ori
        obs_normalized += base_ori_normalized
        # base_lin_vel
        base_lin_vel_normalized = base_lin_vel[::]
        for i in range(0,self._len_base_lin_vel):
            base_lin_vel_normalized[i] = Env0._rescale(base_lin_vel_normalized[i],self.bound_base_lin_vel[i],[-1,1])
        obs += base_lin_vel
        obs_normalized += base_lin_vel_normalized
        # base_ang_vel
        base_ang_vel_normalized = base_ang_vel[::]
        for i in range(0,self._len_base_ang_vel):
            base_ang_vel_normalized[i] = Env0._rescale(base_ang_vel_normalized[i],self.bound_base_ang_vel[i],[-1,1])
        obs += base_ang_vel
        obs_normalized += base_ang_vel_normalized
        return obs, obs_normalized

    def getReward(self):
        reward = 0.
        # Keep the robot standing (fixed base position on Z)
        base_pos, _ = self.robot.getBasePosOri()
        #print("HEIGHT_ROOT: ",HEIGHT_ROOT," and z base: ",base_pos[2])
        reward = 1.0 - abs( HEIGHT_ROOT - base_pos[2] ) # Positive reward
        return reward

    def checkDoneCondition(self):
        done = False
        # Keep the robot root above a treshold
        base_pos, _ = self.robot.getBasePosOri()
        if base_pos[2]<TRESHOLD_DEAD[0] or base_pos[2]>TRESHOLD_DEAD[1]:
            done=True
            print("Episode done, threshold: ",TRESHOLD_DEAD," and position z: ",base_pos[2])
        return done

    # ======================================================================================

    # This needs to be optimized in the future.
    def unnormalizeAction(self, action_normalized):
        action = action_normalized[::]
        number_controlled_joints = len(self.robot.controlled_joints)
        # unnormalize q_des
        for i in range(0,number_controlled_joints):
            action[i] = Env0._rescale( action[i], [-1,1], self.robot.joints_bound_pos[i] ) / DIVIDE_ACTION_POS_VALUE
        # if activated, unnormalize v_des
        if not Q_DESIRED_ONLY:
            for i in range(0,number_controlled_joints):
                action[i+number_controlled_joints] = Env0._rescale( action[i+number_controlled_joints], [-1,1], self.robot.joints_bound_pos[i] )
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

    @staticmethod
    def _run_test_env():
        env = Env0(GUI=True)
        action = np.array([0.01]*env.action_dim)
        while True:
            obs, reward, done, _ = env.step( action )
            if done:
                input("Press to restart ...")
                env.reset()
        pass
    
    @staticmethod
    def _run_test_talos():
        from Robots.ressources.talos import Talos
        Talos._run_test()
        pass

    @staticmethod
    def _run_test_solo():
        from Robots.ressources.solo import Solo
        Solo._run_test()
        pass

    # ======================================= OTHER TESTS

    @staticmethod
    def _run_test_reset_solo():
        from Robots.ressources.solo import Solo
        Solo._run_test_reset()
        pass

    @staticmethod
    def _run_test_reset_talos():
        from Robots.ressources.talos import Talos
        Talos._run_test_reset()
        pass
