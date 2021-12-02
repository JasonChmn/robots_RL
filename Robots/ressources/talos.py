import pybullet as p
import numpy as np
import os
import pybullet_data
import example_robot_data
import time

PATH_URDF = "/opt/openrobots/share/example-robot-data/robots/talos_data/robots"

HIGH_GAINS = True

if HIGH_GAINS:
    FREQUENCY_TALOS_HZ  = 2000          # 5 khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    MULTIPLY_ALL_GAINS_P = 10.
    MULTIPLY_ALL_GAINS_D = 1.
else:
    FREQUENCY_TALOS_HZ  = 2000          # 2khz
    DT = 1/FREQUENCY_TALOS_HZ
    FREQUENCY_UPDATE_CONTROL_HZ  = 50   # 50hz
    DT_PD = 1/FREQUENCY_UPDATE_CONTROL_HZ
    MULTIPLY_ALL_GAINS_P = 1.
    MULTIPLY_ALL_GAINS_D = 1.

GRAVITY = [0,0,-9.81]

class Talos:

    def __init__(self, class_terrain, GUI=False):
        # Client : GUI (Graphical User Interface to check your results) / SHARED_MEMORY (for training with multiprocessing)
        if GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        # Pybullet settings
        p.setTimeStep(DT)
        p.setGravity (GRAVITY[0], GRAVITY[1], GRAVITY[2])
        # Load terrain
        self.terrain = class_terrain()
        # Robot positon and orientation at init
        self.robot_start_position    = [0.0, 0.0, 1.09]  # 1.08]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # Load Talos
        p.setAdditionalSearchPath(PATH_URDF)
        self._robot_ID = p.loadURDF("talos_reduced.urdf",self.robot_start_position, self.robot_start_orientation, useFixedBase=False)
        # Joints indices
        self.controlled_joints, self.all_joints_names, self.all_joints = self._getcontrolled_joints() # indices of controlled joints
        # Joints bounds : Pos and Vel
        self.joints_bound_pos_all, self.joints_bound_vel_all = self._getJointsLimitPosVel(self.all_joints)   # Limit of all joints Pos and Vel   
        self.joints_bound_pos, self.joints_bound_vel = self._getJointsLimitPosVel(self.controlled_joints)    # Limit of controlled joints Pos and Vel
        # Gains PD of all joints
        self.gains_P, self.gains_D                       = self._getGainsPD(self.all_joints)
        self.gains_P_controlled, self.gains_D_controlled = self._getGainsPD(self.controlled_joints)
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
                                          self.robot_start_position,
                                          self.robot_start_orientation, 
                                          self.client)
        p.resetBaseVelocity(self._robot_ID, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], self.client)
        for i in self.controlled_joints:
            p.resetJointState(self._robot_ID, i, 0.0, 0.0, self.client)
        # Reset motors
        no_action = [0.0 for m in self.controlled_joints]
        p.setJointMotorControlArray(self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.VELOCITY_CONTROL, 
                                    targetVelocities = no_action, forces = no_action)
        p.setJointMotorControlArray (self._robot_ID, jointIndices = self.controlled_joints, controlMode = p.TORQUE_CONTROL, forces = no_action)
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
        tau_pd = self.gains_P_controlled * (q_des - q_mes) * MULTIPLY_ALL_GAINS_P + self.gains_D_controlled * (v_des - v_mes) * MULTIPLY_ALL_GAINS_D
        return tau_pd


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
            "gripper_left_joint",
            "gripper_left_inner_double_joint",
            "gripper_left_fingertip_1_joint",
            "gripper_left_fingertip_2_joint",
            "gripper_left_motor_single_joint",
            "gripper_left_inner_single_joint",
            "gripper_left_fingertip_3_joint",
            "wrist_right_ft_joint",  # RIGHT GRIPPER
            "wrist_right_tool_joint",
            "gripper_right_base_link_joint",
            "gripper_right_joint",
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

    @staticmethod
    def _run_test():
        from Robots.ressources.plane import Plane
        robot = Talos(Plane, GUI=True)
        #talos.printControlledJoints()
        q_des = np.array([0.]*len(robot.controlled_joints))
        print("Len : ",len(q_des))
        # Default position (crouch)
        q_des = [   0.00000e+00, 6.76100e-03, # Torso
                    0.00000e+00, 0.00000e+00, # Head
                    2.58470e-01,1.73046e-01,-2.00000e-04,-5.25366e-01,0.00000e+00,0.00000e+00,1.00000e-01, # Left arm
                    -2.58470e-01,-1.73046e-01, 2.00000e-04,-5.25366e-01, 0.00000e+00, 0.00000e+00, 1.00000e-01, # Right arm
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03, # Left leg
                    0.00000e+00, 0.00000e+00,-4.11354e-01, 8.59395e-01,-4.48041e-01,-1.70800e-03  # Right leg
                ]
        print("Len : ",len(q_des))
        v_des = np.array([0.]*len(robot.controlled_joints))
        i = 0
        counter_reset = 300
        while True:
            robot.moveRobot(q_des, v_des, real_time=True, printInfos=True)
            i+=1
            print(i,"/",counter_reset)
            if i%counter_reset==0: 
                robot.reset()
                i=0
    

