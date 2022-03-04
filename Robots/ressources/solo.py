import pybullet as p
import numpy as np
import os
import pybullet_data
import example_robot_data
import time

PATH_URDF = "/opt/openrobots/share/example-robot-data/robots/solo_description/robots"
SOLO_NAME = "solo.urdf"      # This is solo8  : 2 motors per leg only and no shoulder
#SOLO_NAME = "solo12.urdf"    # This is solo12 : 2 motors per leg + all shoulders

HIGH_GAINS = False

FORCE_CONTROLLED_JOINTS_BOUND = True

MAX_TORQUES = 8.0
DIVIDE_BOUNDS_TORQUES = 3.0 # Used when control in torques, lower the torques bounds for controlled joints => To tune

if HIGH_GAINS:
    FREQUENCY_SOLO_HZ  = 5000           # 5 khz
    DT = 1/FREQUENCY_SOLO_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    GAINS_P_ALL = 15.0  # Gains are the same for every joints
    GAINS_D_ALL = 0.3
else:
    FREQUENCY_SOLO_HZ  = 1000           # 1khz
    DT = 1/FREQUENCY_SOLO_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    GAINS_P_ALL = 3.0  # Gains are the same for every joints
    GAINS_D_ALL = 0.07

class Solo:

    """
    ======= INFO ========
    solo8  : 4 legs with 2 motors each (array of size 8)
    solo12 : 4 legs with 2 motors each + 4 shoulders (array of size 12)

    We control the robot at 1Khz (dt=1e-3), but we can run our RL at 50hz for exemple (as done for the MPC).
    
    """

    def __init__(self, class_terrain, GUI=False):
        # Client : GUI (Graphical User Interface to check your results) / SHARED_MEMORY (for training with multiprocessing)
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        # Load terrain
        self.terrain = class_terrain()
        # Load robot
        self._robot_start_pos = [0,0,0.5]
        self._robot_start_orientation = p.getQuaternionFromEuler([0,0,0])
        p.setAdditionalSearchPath(PATH_URDF)
        self._robot_ID = p.loadURDF(SOLO_NAME, self._robot_start_pos, self._robot_start_orientation, useFixedBase=False)
        # Parameters
        # - Timestep: We control the robot at 1Khz (PD controller) and we give a new control(q_des,v_des) at 50 hz.
        p.setTimeStep(DT, self.client)
        # - gravity
        p.setGravity(0, 0, -9.81)
        # - Get joints to control (with motors)
        non_controlled_joints = [ "FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE" ] # These joints are just used to get info (pos/vel/ori?)
        self.all_joints_name = [p.getJointInfo(self._robot_ID, i)[1].decode() for i in range(p.getNumJoints(self._robot_ID))] # Name of all joints
        self.all_joints   = [i for i in range(p.getNumJoints(self._robot_ID))] # Indices of all joints : 12 for solo8 / 16 for solo 12
        self.controlled_joints = [i for (i, n) in enumerate(self.all_joints_name) if n not in non_controlled_joints] # Indices of controlled joints
        # Force controlled joints bound
        _HAA_bounds = [-1,1]
        _HFE_bounds = [-1.5,1.5]
        _KFE_front_bounds = [-2.5,0.5]
        _KFE_back_bounds  = [-0.5,2.5]
        if SOLO_NAME=="solo12.urdf":
            self.controlled_joints_bound = [_HAA_bounds, _HFE_bounds,_KFE_front_bounds,
                                            _HAA_bounds, _HFE_bounds,_KFE_front_bounds,
                                            _HAA_bounds, _HFE_bounds,_KFE_back_bounds,
                                            _HAA_bounds, _HFE_bounds,_KFE_back_bounds ]
        else:
            self.controlled_joints_bound = [_HFE_bounds,_KFE_front_bounds,
                                            _HFE_bounds,_KFE_front_bounds,
                                            _HFE_bounds,_KFE_back_bounds,
                                            _HFE_bounds,_KFE_back_bounds ]
        # - Gains
        self.gains_P = [GAINS_P_ALL]*len(self.controlled_joints)
        self.gains_D = [GAINS_D_ALL]*len(self.controlled_joints)
        # Joints bounds : Pos and Vel
        self.joints_bound_pos_all, self.joints_bound_vel_all = self._getJointsLimitPosVel(self.all_joints)   # Limit of all joints Pos and Vel   
        self.joints_bound_pos, self.joints_bound_vel = self._getJointsLimitPosVel(self.controlled_joints)    # Limit of controlled joints Pos and Vel
        # Controlled joints bounds : torques
        self.joints_bound_torques = self._get_max_torques_joints(self.controlled_joints)
        # Reset robot
        self.reset()
        # - printInfos
        print("=== SOLO CREATED")
        print("Number of controlled_joints : ",len(self.controlled_joints)," / ",len(self.all_joints))
        # Options (to delete)
        self.info_max_torques = 0.0
        pass

    def __del__(self):
        p.disconnect()
        pass

    # Reset the position of the robot. You can modify self._robot_start_pos if you want.
    def reset(self):
        p.resetBasePositionAndOrientation(self._robot_ID, self._robot_start_pos,
                                          self._robot_start_orientation, self.client)
        p.resetBaseVelocity(self._robot_ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], self.client)
        for i in self.controlled_joints:
            p.resetJointState(self._robot_ID, i, 0.0, 0.0, self.client)

        # Reset motors
        no_action = [0.0 for m in self.controlled_joints]
        p.setJointMotorControlArray(self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.VELOCITY_CONTROL, 
                                    targetVelocities = no_action, forces = no_action)
        p.setJointMotorControlArray (self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.TORQUE_CONTROL, forces = no_action)
        pass

    # =========================================================================================================

    # ================================================================

    # - All joints
    # Get angle position and velocity of each joint.
    # @output 
    # - q_mes : list of joint angles (in radians)
    # - v_mes : list of joint angular velocities (in radians/sec)
    def getJointsState(self):
        q_mes, v_mes = self._getJointsState(self.all_joints)  # State of all joints
        return q_mes, v_mes
    # Get joints bound : Position (angle) and Velocity (angular vel)
    def getJointsBounds(self):
        return self.joints_bound_pos_all, self.joints_bound_vel_all

    # - Controlled joints
    # Get controlled joints state : Position (angle) and Velocity (angular vel)
    # @output 
    # - q_mes : list of joint angles (in radians)
    # - v_mes : list of joint angular velocities (in radians/sec)
    def getControlledJointsState(self):
        q_mes, v_mes = self._getJointsState(self.controlled_joints)
        return q_mes, v_mes
    # Get joints bound : Position (angle) and Velocity (angular vel)
    def getControlledJointsBounds(self):
        q_bounds, v_bounds = self.joints_bound_pos, self.joints_bound_vel
        return q_bounds, v_bounds
    # Get joints bbound : Torques
    def getControlledJointsTorquesBounds(self):
        return self.joints_bound_torques

    # - Base
    # Get position and orientation of the base of the robot.
    # @output
    # - base_pos : 3D position
    # - base_orientation : Quaternion (4 values)
    def getBasePosOri(self):
        b_pos, b_orientation = p.getBasePositionAndOrientation(self._robot_ID)
        base_pos = [value for value in b_pos]
        base_orientation = [value for value in b_orientation]
        return base_pos, base_orientation
    # Get velocity of the base of the robot.
    # @output
    # - lin_vel : linear velocity
    # - ang_vel : angular velocity
    def getBaseVel(self):
        l_vel, a_vel = p.getBaseVelocity(self._robot_ID)
        lin_vel = [value for value in l_vel]
        ang_vel = [value for value in a_vel]
        return lin_vel, ang_vel
    # [Info] There are no bounds for Base, you need to define it in your environment.

    # =========================================================================================================



    # Move joints of the robot to desired positon and velocity
    # @input
    # - q_des : list of desired joint angles
    # - v_des : list of desired joint angular velocities
    # - dt_controller : timestep of the controller.
    #                   If we send the command to the robot at 1Khz (dt=1e-3) and send a new "v_des" and "v_mes" at 50hz (dt_controller=1/50=0.02).
    #                   In this function we will send the command : 0.02/1e-3 = 20 times.
    # - real_time : boolean to run the simulation in real time.
    def moveRobot(self, q_des, v_des, real_time=True, printInfos=False):
        time_simulation = 0.0
        while time_simulation<DT_PD:
            if real_time: 
                t_start = time.time()
            # Compute torques
            q_mes, v_mes = self._getJointsState(self.controlled_joints)
            torques = self._computePDTorques(np.array(q_des), np.array(q_mes), np.array(v_des), np.array(v_mes)).tolist()
            #for t in torques:
            #    if t>self.info_max_torques: self.info_max_torques=t
            if printInfos:
                print("------")
                print("q_mes: ",np.round(q_mes,4))
                print("q_des: ",np.round(q_des,4))
                print("torques : ",np.round(torques))
            # Apply torques on robot
            p.setJointMotorControlArray(self._robot_ID, self.controlled_joints,
                                        controlMode=p.TORQUE_CONTROL, forces=torques) # There is another function if we run it on the real robot.
            #q_mes, v_mes = self._getJointsState(self.controlled_joints)
            # Increment time
            time_simulation += DT
            # Run simulation
            p.stepSimulation()
            # Wait if real time
            if real_time:
                while (time.time() - t_start) < DT:
                    pass
        return None

    def moveRobot_torques(self, torques, real_time=True, printInfos=False):
        time_simulation = 0.0
        if printInfos:
            print("torques : ",np.round(torques,1))
        # "Constant torques are applied for the duration of a control step" as in : https://arxiv.org/pdf/1611.01055.pdf
        while time_simulation<DT_PD:
            if real_time: 
                t_start = time.time()
            # Apply torques on robot
            p.setJointMotorControlArray(self._robot_ID, self.controlled_joints,
                                        controlMode=p.TORQUE_CONTROL, forces=torques) # There is another function if we run it on the real robot.
            #q_mes, v_mes = self._getJointsState(self.controlled_joints)
            # Increment time
            time_simulation += DT
            # Run simulation
            p.stepSimulation()
            # Wait if real time
            if real_time:
                while (time.time() - t_start) < DT:
                    pass
        return None

    # Compute PD torques.
    # @input
    # - q_des : list of desired joint angles
    # - q_mes : list of mesured joint angles
    # - v_des : list of desired joint angular velocities
    # - v_mes : list of mesured joint angular velocities
    # @output
    # - torques : torques to apply on joints
    def _computePDTorques(self, q_des, q_mes, v_des, v_mes):
        torques = self.gains_P * (q_des - q_mes) + self.gains_D * (v_des - v_mes)
        torques = np.array([min(t,MAX_TORQUES) for t in torques])
        return torques


    # - List of joints
    # Get angle position and velocity of each joint.
    # @output 
    # - q_mes : list of joint angles (in radians)
    # - v_mes : list of joint angular velocities (in radians/sec)
    def _getJointsState(self, listJoints):
        joint_states = p.getJointStates(self._robot_ID, listJoints)  # State of all joints
        q_mes = [state[0] for state in joint_states]
        v_mes = [state[1] for state in joint_states]
        return q_mes, v_mes

    def _getJointsLimitPosVel(self, joints_indices):
        print("===")
        joints_bound_pos, joints_bound_vel = [], []
        for i in joints_indices:
            info = p.getJointInfo(self._robot_ID, i)
            if FORCE_CONTROLLED_JOINTS_BOUND and i in self.controlled_joints:
                joints_bound_pos.append( self.controlled_joints_bound[ self.controlled_joints.index(i) ] )
            else:
                joints_bound_pos.append([info[8], info[9]]) # Pos
            joints_bound_vel.append([-info[11],info[11]])   # Vel
        #for j in joints_bound_pos: print(j)
        #input("...")
        return joints_bound_pos, joints_bound_vel

    # ============================================================================================

    # Get max torques possible for each joint. We do not consider joints velocity.
    # Formula : torques =
    def _get_max_torques_joints(self, joint_indices):
        # Compute list of torques from min to max joint positions
        min_pos_joints = [ self.joints_bound_pos_all[i][0] for i in joint_indices ]
        max_pos_joints = [ self.joints_bound_pos_all[i][1] for i in joint_indices ]
        null_values = [ 0. for _ in joint_indices ]
        max_torques = self._computePDTorques(np.array(min_pos_joints), np.array(max_pos_joints), 
                                             np.array(null_values), np.array(null_values)
                                            ).tolist()
        # Compute torques bounds
        joints_bound_torques = []
        for i in range(len(joint_indices)):
            joints_bound_torques.append( [-max_torques[i]/DIVIDE_BOUNDS_TORQUES, max_torques[i]/DIVIDE_BOUNDS_TORQUES] )
        return joints_bound_torques

    # ============================================================================================

    @staticmethod
    def _run_test():
        from Robots.ressources.plane import Plane
        robot = Solo(Plane, GUI=True)
        if SOLO_NAME=="solo12.urdf":
            q_des = np.array([0.0, 0.7, -1.4, 0.0, 0.7, -1.4, 0.0, -0.7, 1.4, 0.0, -0.7, 1.4]) # Position default for solo 12
        else:
            q_des = np.array([0.7, -1.4, 0.7, -1.4, -0.7, 1.4, -0.7, 1.4]) # Position default for solo 12
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter_reset = 250
        while True:
            robot.moveRobot(q_des, v_des, real_time=True)
            i+=1
            print(i,"/",counter_reset)
            if i%counter_reset==0: 
                robot.reset()
                i=0
        pass

    @staticmethod
    def _run_test_reset():
        from Robots.ressources.plane import Plane
        robot = Solo(Plane, GUI=True)
        if SOLO_NAME=="solo12.urdf":
            q_des = np.array([0.0, 0.7, -1.4, 0.0, 0.7, -1.4, 0.0, -0.7, 1.4, 0.0, -0.7, 1.4]) # Position default for solo 12
        else:
            q_des = np.array([0.7, -1.4, 0.7, -1.4, -0.7, 1.4, -0.7, 1.4]) # Position default for solo 12
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter_reset = 20
        max_tab_check = 15
        tab_states = []
        while True:
            # Test configs
            if len(tab_states)<max_tab_check:
                q, v = robot.getJointsState()
                tab_states.append(q+v)
            else:
                if i<max_tab_check:
                    q, v = robot.getJointsState()
                    state = q+v
                    same = True
                    for j,value_original in enumerate(tab_states[i]):
                        if state[j]!=value_original:
                            same=False
                            break
                    if not same:
                        print("s bef : ",tab_states[i])
                        print("s aft : ",state)
                        input("Not same ...")
            # move robot
            robot.moveRobot(q_des, v_des, real_time=True)
            # Reset
            i+=1
            print(i,"/",counter_reset)
            if i%counter_reset==0: 
                robot.reset()
                i=0
        pass

    @staticmethod
    def _run_test_joints_limit():
        from Robots.ressources.plane import Plane
        robot = Solo(Plane, GUI=True)
        # Move robot to default position
        if SOLO_NAME=="solo12.urdf":
            q_des = np.array([0.0, 0.7, -1.4, 0.0, 0.7, -1.4, 0.0, -0.7, 1.4, 0.0, -0.7, 1.4]) # Position default for solo 12
        else:
            q_des = np.array([0.7, -1.4, 0.7, -1.4, -0.7, 1.4, -0.7, 1.4]) # Position default for solo 9
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter = 50
        done = False
        while not done:
            robot.moveRobot(q_des, v_des, real_time=True)
            i+=1
            if i%counter==0:
                done = True
        # Test bounds of each joint
        input("Start test for each joint ... (press key)")
        for index_joint, bound in enumerate(robot.joints_bound_pos):
            print("==== Joint ",index_joint," : ")
            for j in range(2): # Min and Max
                value = bound[j]
                print("value tested = ",value)
                q_des_aux = q_des.copy()
                q_des_aux[index_joint] = value
                i = 0
                done = False
                while not done:
                    robot.moveRobot(q_des_aux, v_des, real_time=True)
                    i+=1
                    if i%counter==0:
                        done = True
                        input("Press key...")
                # Reset to default position
                i = 0
                done = False
                while not done:
                    robot.moveRobot(q_des, v_des, real_time=True)
                    i+=1
                    if i%counter==0:
                        done = True
        pass