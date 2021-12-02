import pybullet as p
import numpy as np
import os
import pybullet_data
import example_robot_data
import time

PATH_URDF = "/opt/openrobots/share/example-robot-data/robots/solo_description/robots"
SOLO_NAME = "solo.urdf"      # This is solo8  : 2 motors per leg only and no shoulder
SOLO_NAME = "solo12.urdf"    # This is solo12 : 2 motors per leg + all shoulders

HIGH_GAINS = True

if HIGH_GAINS:
    FREQUENCY_TALOS_HZ  = 5000          # 5 khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    GAINS_P_ALL = 40.0  # Gains are the same for every joints
    GAINS_D_ALL = 1.0
else:
    FREQUENCY_TALOS_HZ  = 1000          # 1khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    GAINS_P_ALL = 3.0  # Gains are the same for every joints
    GAINS_D_ALL = 0.2

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
        self.all_joints   = [i for i in range(p.getNumJoints(self._robot_ID))] # Indices of all joints
        self.controlled_joints = [i for (i, n) in enumerate(self.all_joints_name) if n not in non_controlled_joints] # Indices of controlled joints
        # - Gains
        self.gains_P = [GAINS_P_ALL]*len(self.controlled_joints)
        self.gains_D = [GAINS_D_ALL]*len(self.controlled_joints)
        # Joints bounds : Pos and Vel
        self.joints_bound_pos_all, self.joints_bound_vel_all = self._getJointsLimitPosVel(self.all_joints)   # Limit of all joints Pos and Vel   
        self.joints_bound_pos, self.joints_bound_vel = self._getJointsLimitPosVel(self.controlled_joints)    # Limit of controlled joints Pos and Vel
        # Reset robot
        self.reset()
        # - printInfos
        print("=== TALOS CREATED")
        print("Number of controlled_joints : ",len(self.controlled_joints)," / ",len(self.all_joints))
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
        q_bounds, v_bounds = self.joints_bound_pos_all, self.joints_bound_vel_all
        return q_bounds, v_bounds

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
    def moveRobot(self, q_des, v_des, real_time=True, printInfos=True):
        time_simulation = 0.0
        while time_simulation<DT_PD:
            if real_time: 
                t_start = time.time()
            # Compute torques
            q_mes, v_mes = self._getJointsState(self.controlled_joints)
            torques = self._computePDTorques(np.array(q_des), np.array(q_mes), np.array(v_des), np.array(v_mes)).tolist()
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
        pass

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
        joints_bound_pos, joints_bound_vel = [], []
        for i in joints_indices:
            info = p.getJointInfo(self._robot_ID, i)
            joints_bound_pos.append([info[8], info[9]]) # Pos
            joints_bound_vel.append([-info[11],info[11]])   # Vel
        return joints_bound_pos, joints_bound_vel


    @staticmethod
    def _run_test():
        from Robots.ressources.plane import Plane
        robot = Solo(Plane, GUI=True)
        #q_des = np.array([0.05]*len(robot.controlled_joints)) # 0.55 max ?
        q_des = np.array([0.0, 0.7, -1.4, 0.0, 0.7, -1.4, 0.0, -0.7, 1.4, 0.0, -0.7, 1.4]) # Position default
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
    