import pybullet as p
import numpy as np
import os
import pybullet_data
import example_robot_data
import time

PATH_URDF = "/opt/openrobots/share/example-robot-data/robots/talos_data/robots"

HIGH_GAINS = False  # Two setup for GAINS. The high gains are the one used by Gepetto team. But in RL, we have to lower it (keep False) => To tune.

# For MAX_TORQUES=2 and DIVIDE_BOUNDS_TORQUES=4 => The max torques possible will be 2/4=0.5.
MAX_TORQUES = 100. # Knee 300, Legs 160-200, Shoulder 22-44 => To tune.
DIVIDE_BOUNDS_TORQUES = 2.0 # Used with torque control, lower the torques bounds for controlled joints => To tune.

if HIGH_GAINS:
    # Parameters used in GEPETTO
    FREQUENCY_TALOS_HZ  = 5000          # 5 khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    MULTIPLY_ALL_GAINS_P = 10.
    MULTIPLY_ALL_GAINS_D = 1.
else:
    # Parameters used for RL
    FREQUENCY_TALOS_HZ  = 2000          # 2khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    MULTIPLY_ALL_GAINS_P = 1.
    MULTIPLY_ALL_GAINS_D = 1.
    if Config.MODE_CONTROL=="TORQUE":
        # Paper : "A Comparison of Action Spaces for Learning Manipulation Tasks"
        # The policies were queried at 10Hz, and the low-level controllers operated at 100Hz across all experiments.
        # READ THIS : https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0383251
        # We do the same.
        FREQUENCY_TALOS_HZ  = 200
        DT = 1/FREQUENCY_SOLO_HZ
        FREQUENCY_UPDATE_CONTROL_HZ  = 200
        DT_TORQUES = 1/FREQUENCY_UPDATE_CONTROL_HZ

class Talos:

    def __init__(self, class_terrain, GUI=False, useFixedBase=False):
        # Client : GUI (Graphical User Interface to check your results) / SHARED_MEMORY (for training with multiprocessing)
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        # Pybullet settings
        p.setTimeStep(DT)
        p.setGravity(0, 0, -9.81)
        # Load terrain
        self.terrain = class_terrain()
        # Robot positon and orientation at init
        self._robot_start_pos    = [0.0, 0.0, 1.09]  # 1.08]
        self._robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # Load Talos
        p.setAdditionalSearchPath(PATH_URDF)
        self._robot_ID = p.loadURDF("talos_reduced.urdf",self._robot_start_pos, self._robot_start_orientation, useFixedBase=useFixedBase)
        # Joints indices
        self.controlled_joints, self.all_joints_names, self.all_joints = self._getcontrolled_joints() # indices of controlled joints
        self.non_controlled_joints_to_reset = [21,38] # gripper_left_joint and gripper_right_joint
        # Joints bounds : Pos and Vel
        self.joints_bound_pos_all, self.joints_bound_vel_all = self._getJointsLimitPosVel(self.all_joints)   # Limit of all joints Pos and Vel   
        self.joints_bound_pos, self.joints_bound_vel = self._getJointsLimitPosVel(self.controlled_joints)    # Limit of controlled joints Pos and Vel
        # Gains PD of all joints
        self.gains_P, self.gains_D                       = self._getGainsPD(self.all_joints)
        self.gains_P_controlled, self.gains_D_controlled = self._getGainsPD(self.controlled_joints)
        # Controlled joints bounds : torques
        self.joints_bound_torques = self._get_max_torques_joints(self.controlled_joints)
        # Set motors control
        no_action = [0.0 for m in self.controlled_joints]
        p.setJointMotorControlArray(self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.VELOCITY_CONTROL, 
                                    targetVelocities = no_action, forces = no_action)
        p.setJointMotorControlArray (self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.TORQUE_CONTROL, forces = no_action)
        # Reset robot
        self.reset()
        # - printInfos
        print("=== SOLO CREATED")
        print("Number of controlled_joints : ",len(self.controlled_joints)," / ",len(self.all_joints))
        pass

    def __del__(self):
        p.disconnect()
        pass

    # Reset position, orientation, velocity, torques of the robot.
    def reset(self):
        # Reset base
        p.resetBasePositionAndOrientation(self._robot_ID, 
                                          self._robot_start_pos,
                                          self._robot_start_orientation, 
                                          self.client)
        p.resetBaseVelocity(self._robot_ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], self.client)
        # Reset joints
        for i in self.controlled_joints:
            p.resetJointState(self._robot_ID, i, 0.0, 0.0, self.client)
        # For this robot, we do not control the grippers, but we need to reset it
        for i in self.non_controlled_joints_to_reset:
            p.resetJointState(self._robot_ID, i, 0.0, 0.0, self.client)
        pass


    # Move joints of the robot to desired positon and velocity
    # @input
    # - q_des : list of desired joint angles
    # - v_des : list of desired joint angular velocities
    # - dt_controller : timestep of the controller.
    #                   If we send the command to the robot at 1Khz (dt=1e-3) and send a new "v_des" and "v_mes" at 50hz (dt_controller=1/50=0.02).
    #                   In this function we will send the command : 0.02/1e-3 = 20 times.
    # - real_time : boolean to run the simulation in real time.
    def moveRobot(self, q_des, v_des, real_time=False, printInfos=False):
        # Check if arguments are valid
        if len(self.controlled_joints)!=len(q_des) or len(q_des)!=len(v_des):
            print("controlled_joints:",len(self.controlled_joints),", q_des:",len(q_des)," v_des:",len(v_des))
            input("ERROR, q_des/v_des/controlled_joints have different sizes")
        # Run simulation
        time_simulation = 0.0
        while time_simulation<DT_PD:
            if real_time: 
                t_start = time.time()
            # Compute torques
            q_mes, v_mes = self._getJointsState(self.controlled_joints)
            torques = self._computePDTorques(np.array(q_des), np.array(q_mes), np.array(v_des), np.array(v_mes))
            if printInfos:
                print("------")
                print("q_mes: ",np.round(q_mes,4))
                print("q_des: ",np.round(q_des,4))
                print("torques : ",np.round(torques))
            # Apply torques on robot
            p.setJointMotorControlArray(self._robot_ID, self.controlled_joints,
                                        controlMode=p.TORQUE_CONTROL, forces=torques) # There is another function if we run it on the real robot.
            # Increment time
            time_simulation += DT
            # Run simulation
            p.stepSimulation()
            # Wait if real time
            if real_time:
                while (time.time() - t_start) < DT:
                    pass
        pass

    def moveRobot_torques(self, torques, real_time=True, print_info=False):
        time_simulation = 0.0
        if print_info:
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

    # ============================================================================================

    # Get max torques possible for each joint. We do not consider joints velocity.
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
            joints_bound_torques.append( [-abs(max_torques[i])/DIVIDE_BOUNDS_TORQUES, abs(max_torques[i])/DIVIDE_BOUNDS_TORQUES] )
        #print(joints_bound_torques)
        return joints_bound_torques

    # ================================================================

    # - All joints
    # Get angle position and velocity of each joint.
    # @output 
    # - q_mes : list of joint angles (in radians)
    # - v_mes : list of joint angular velocities (in radians/sec)
    def getJointsState(self):
        q_mes, v_mes = self._getJointsState(self.all_joints)
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
    # Get joints bound : Torques
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
    

    # ====================================================================================================================

    def printControlledJoints(self):
        print("List of controlled joints:")
        for i in self.controlled_joints:
            print(" - ",self.all_joints_names[i])
            print("   Gains: P=",self.gains_P[i]," D=",self.gains_D[i])
        pass

    # ====================================================================================================================


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
        torques = np.array([np.copysign(min(abs(t),MAX_TORQUES),t) for t in torques])
        #print(torques)
        return torques


    # ===================================================================================================================

    def _getcontrolled_joints(self):
        non_controlled_joints = [
            #"torso_1_joint",# Other Body Joints
            #"torso_2_joint",
            #"head_1_joint",
            #"head_2_joint",
            #"arm_left_1_joint",#Left Joints
            #"arm_left_2_joint",
            #"arm_left_3_joint",
            #"arm_left_4_joint",
            #"arm_left_5_joint",
            #"arm_left_6_joint",
            #"arm_left_7_joint",
            #"arm_right_1_joint",#Right Joints
            #"arm_right_2_joint",
            #"arm_right_3_joint",
            #"arm_right_4_joint",
            #"arm_right_5_joint",
            #"arm_right_6_joint",
            #"arm_right_7_joint",
            #"leg_left_1_joint", #Left Leg
            #"leg_left_2_joint",
            #"leg_left_3_joint",
            #"leg_left_4_joint",
            #"leg_left_5_joint",
            #"leg_left_6_joint",
            #"leg_right_1_joint", #right Leg
            #"leg_right_2_joint",
            #"leg_right_3_joint",
            #"leg_right_4_joint",
            #"leg_right_5_joint",
            #"leg_right_6_joint",
            "imu_joint",  # Other joints not used
            "rgbd_joint",
            "rgbd_optical_joint",
            "rgbd_depth_joint",
            "rgbd_depth_optical_joint",
            "rgbd_rgb_joint",
            "rgbd_rgb_optical_joint",
            "wrist_left_ft_joint",  # LEFT GRIPPER
            "wrist_left_tool_joint",
            "gripper_left_base_link_joint",
            "gripper_left_joint", # This is controllable
            "gripper_left_inner_double_joint",
            "gripper_left_fingertip_1_joint",
            "gripper_left_fingertip_2_joint",
            "gripper_left_motor_single_joint",
            "gripper_left_inner_single_joint",
            "gripper_left_fingertip_3_joint",
            "wrist_right_ft_joint",  # RIGHT GRIPPER
            "wrist_right_tool_joint",
            "gripper_right_base_link_joint",
            "gripper_right_joint", # This is controllable
            "gripper_right_inner_double_joint",
            "gripper_right_fingertip_1_joint",
            "gripper_right_fingertip_2_joint",
            "gripper_right_motor_single_joint",
            "gripper_right_inner_single_joint",
            "gripper_right_fingertip_3_joint",
            "leg_left_sole_fix_joint",  # LEFT LEG (blocked)
            "leg_right_sole_fix_joint"  # RIGHT LEG (blocked)
        ]
        all_joints_names   = [p.getJointInfo(self._robot_ID, i)[1].decode() for i in range(p.getNumJoints(self._robot_ID))]  # Name of all joints
        all_joints = [i for i in range(0, len(all_joints_names))]
        controlled_joints  = [i for (i, n) in enumerate(all_joints_names) if n not in non_controlled_joints]
        """
        print("Joints controlled : ")
        for i in controlled_joints:
            print(i," - ",all_joints_names[i])
        """
        return controlled_joints, all_joints_names, all_joints

    def _getJointsLimitPosVel(self, joints_indices):
        joints_bound_pos, joints_bound_vel = [], []
        for i in joints_indices:
            info = p.getJointInfo(self._robot_ID, i)
            joints_bound_pos.append(info[8:10]) # Pos
            joints_bound_vel.append([-info[11],info[11]])   # Vel
        return joints_bound_pos, joints_bound_vel

    def _getGainsPD(self, joints_indices):
        gains_P, gains_D = [], []
        # These gains are from another work on TALOS. This may need to be modified.
        gains_P_complete = np.array([
                200.,  200.,                                   # TORSO 0-1
                30., 30.,                                      # HEAD 2-3
                200.,  400.,  100.,  100.,  10.,  10.,  10.,   # LEFT ARM 4-10
                200.,  400.,  100.,  100.,  10.,  10.,  10.,   # RIGHT ARM 11-17
                100.,  100.,  100.,  100.,  100.,  100.,       # LEFT LEG 18-23
                100.,  100.,  100.,  100.,  100.,  100.,       # RIGHT LEG 24-29
            ])
        """
        gains_D_complete = np.array([
                10.,  10.,                          # TORSO 0-1
                0.1, 0.1,                           # HEAD 2-3
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # LEFT ARM 4-10
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # RIGHT ARM 11-17
                20.,  20.,  20.,  20.,  20.,  20.,  # LEFT LEG 18-23
                20.,  20.,  20.,  20.,  20.,  20.,  # RIGHT LEG 24-29
            ])
        """
        gains_D_complete = np.array([
                1.,  1.,                          # TORSO 0-1
                0.1, 0.1,                           # HEAD 2-3
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # LEFT ARM 4-10
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # RIGHT ARM 11-17
                20.,  20.,  20.,  20.,  20.,  20.,  # LEFT LEG 18-23
                20.,  20.,  20.,  20.,  20.,  20.,  # RIGHT LEG 24-29
            ])
        # Map these gains to controlled joints
        # Mapping done with this :
        """
        for i,joint in enumerate(self.all_joints_names):
            print(i," - ",self.all_joints_names[i])
        """
        map_gains_to_joints = [
                                0,1,                    # TORSO 0-1
                                3,4,                    # HEAD 2-3
                                11,12,13,14,15,16,17,   # LEFT ARM 4-10
                                28,29,30,31,32,33,34,   # RIGHT ARM 11-17
                                45,46,47,48,49,50,      # LEFT LEG 18-23
                                52,53,54,55,56,57       # RIGHT LEG 24-29
                              ]
        map_joints_to_gains = [
                                            0,1,    # Torso 0-1
                                            None,   # imu 2
                                            2,3,    # Head  3-4
                                            None, None, None, None, None, None, # rgbd 6-9
                                            4,5,6,7,8,9,10,       # Left arm  11-17
                                            None, None, None, None, None, None, None, None, None, None, # wrist / gripper left  18-27
                                            11,12,13,14,15,16,17, # Right arm 28-34
                                            None, None, None, None, None, None, None, None, None, None, # wrist / gripper right 35-44
                                            18,19,20,21,22,23,  # Left leg   45-50
                                            None, # leg_left_sole_fix_joint  51
                                            24,25,26,27,28,29,  # Right leg  52-57
                                            None # right_left_sole_fix_joint 52
                              ]

        # Get gains of joints in parameter
        for i in joints_indices:
            index_in_gains = map_joints_to_gains[i]
            gains_P.append( gains_P_complete[index_in_gains] )
            gains_D.append( gains_D_complete[index_in_gains] )
        return gains_P, gains_D

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

    # ============================================================================================

    # TEST : Run the simulation and move the robot to default position.
    @staticmethod
    def _run_test():
        from Robots.ressources.plane import Plane
        robot = Talos(Plane, GUI=True)
        q_des = np.array([0.]*len(robot.controlled_joints))
        # Default position (crouch)
        q_des = [   0.00000e+00, 6.76100e-03, # Torso
                    0.00000e+00, 0.00000e+00, # Head
                    2.58470e-01,1.73046e-01,-2.00000e-04,-5.25366e-01,0.00000e+00,0.00000e+00,1.00000e-01, # Left arm
                    -2.58470e-01,-1.73046e-01, 2.00000e-04,-5.25366e-01, 0.00000e+00, 0.00000e+00, 1.00000e-01, # Right arm
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03, # Left leg
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03  # Right leg
                ]
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter_reset = 300
        while True:
            robot.moveRobot(q_des, v_des, real_time=True, printInfos=False)
            i+=1
            print(i,"/",counter_reset)
            if i%counter_reset==0: 
                robot.reset()
                i=0
    

    # TEST : Run the simulation and test the reset.
    @staticmethod
    def _run_test_reset():
        from Robots.ressources.plane import Plane
        robot = Talos(Plane, GUI=True)
        q_des = np.array([0.]*len(robot.controlled_joints))
        # Default position (crouch)
        q_des = [   0.00000e+00, 6.76100e-03, # Torso
                    0.00000e+00, 0.00000e+00, # Head
                    2.58470e-01,1.73046e-01,-2.00000e-04,-5.25366e-01,0.00000e+00,0.00000e+00,1.00000e-01, # Left arm
                    -2.58470e-01,-1.73046e-01, 2.00000e-04,-5.25366e-01, 0.00000e+00, 0.00000e+00, 1.00000e-01, # Right arm
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03, # Left leg
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03  # Right leg
                ]
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter_reset = 100
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
                        index = j
                        if state[index]!=value_original:
                            print("different : ",robot.all_joints_names[index%len(q)])
                            same=False
                            input("...")
                    if not same:
                        print("Last different:")
                        print("  q: ",q)
                        print("   : ",tab_states[i][0:len(q)])
                        print("  v: ",v)
                        print("   : ",tab_states[i][len(q):-1])
                        input("Not same ...")
            # Move robot
            robot.moveRobot(q_des, v_des, real_time=True, printInfos=False)
            # Reset
            i+=1
            print(i,"/",counter_reset)
            if i%counter_reset==0: 
                robot.reset()
                i=0
        pass


    # TEST : Show the limit of each joints one by one (bounds: Robot.joints_bound_pos).
    @staticmethod
    def _run_test_joints_limit():
        from Robots.ressources.plane import Plane
        robot = Talos(Plane, GUI=True, useFixedBase=True)
        # Move robot to default position
        q_des = [   0.00000e+00, 6.76100e-03, # Torso
                    0.00000e+00, 0.00000e+00, # Head
                    2.58470e-01,1.73046e-01,-2.00000e-04,-5.25366e-01,0.00000e+00,0.00000e+00,1.00000e-01, # Left arm
                    -2.58470e-01,-1.73046e-01, 2.00000e-04,-5.25366e-01, 0.00000e+00, 0.00000e+00, 1.00000e-01, # Right arm
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03, # Left leg
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03  # Right leg
                ]
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter = 50
        done = False
        while not done:
            robot.moveRobot(q_des, v_des, real_time=True, printInfos=False)
            i+=1
            if i%counter==0: 
                done = True
        # Test bounds of each joint
        input("Start test for each joint ... (press key)")
        for index_joint, bound in enumerate(robot.joints_bound_pos):
            print("==== Joint ",index_joint," : ", robot.all_joints_names[robot.controlled_joints[index_joint]])
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