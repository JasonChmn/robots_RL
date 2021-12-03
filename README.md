# Robots_RL
Classes for Solo and Talos Robot + Exemple of RL using PPO2 (stableBaseline2).
https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

The actual RL reward encourages the robot to keep his root position to a fix height.
It is a very simple reward that may not work for now, and would require to tune gains/parameters in the robot class and rewards/states/action in the environment.

##Solo and Talos classes
Classes to control Solo and Talos robots in pybullet with a simple PD controller.
Robots URDF : https://github.com/Gepetto/example-robot-data/tree/master/robots in talos_data and solo_description.

You can select the robot in env0.py, variable NAME_ROBOT.
Modify all the parameters in env0.py in function of the RL task to perform.

### Variables
- **[TO SET] ``PATH_URDF`` : Path to the URDF file of the robot (You need to set it for each robot).**
- ``HIGH_GAINS`` : Gains and Frequencies of the robots need to be tuned to be used with RL. 
               if False, we use the default gains and frequencies in Gepetto's experiment used with feedforward couples.
               if True, we set higher gains and frequencies to use it with RL (without feedforward couples).
- ``FREQUENCY_TALOS_HZ``, ``GAINS_P_ALL``, ``GAINS_D_ALL`` : These values need to be tuned if the robot does not have enough strength (can not move) or his actions are jerky and unstable. You can test different combination of gains and frequencies with the class test_PD.py
- ``FREQUENCY_UPDATE_CONTROL_HZ`` : Frequency at which we update q_des and v_des on the PD controller. 50 hz is a very low value and we may need to increase it.
- (For Solo) ``SOLO_NAME`` : Select solo8 or solo12. Solo12 has 4 more controlled joints on shoulders.
                                                 
### Functions
- ``reset()`` : Reset the robot position/orientation and forces applied on it. The robot's starting configuration is defined by parameters ``_robot_start_pos`` and ```_robot_start_orientation``.
- ``moveRobot()`` : Use the PD controller on the robot for q_des qnd v_des. 
                Each call of this function runs the PD controller for 1/FREQUENCY_UPDATE_CONTROL_HZ seconds.
                The PD controller computes the torques and apply it on the robot every 1/FREQUENCY_TALOS_HZ seconds.
                Example for FREQUENCY_UPDATE_CONTROL_HZ=2000 and FREQUENCY_UPDATE_CONTROL_HZ=50 :
                moveRobot() for 1/50=0.02s with an update of torques every 1/2000=0.0005sec.
- ``getJointsState()``  : Return q_mes and v_mes of all joints.
- ``getJointsBounds()`` : Return bounds for q_mes and v_mes of all joints.
- ``getControlledJointsState()``  : Return q_mes and v_mes of controlled joints.
- ``getControlledJointsBounds()`` : Return bounds for q_mes and v_mes of controlled joints.
- ``getBasePosOri()`` : Return base position and orientation.
- ``getBaseVel()``    : Return base linear and angular velocity.

###Examples
All these examples can be found in the main of respective classes.
The code to run the RL algorithm: to learn a new policy and play it can be found in run_env.py.
The hyperparameters used for PPO need to be tuned.

## Solo
This test can be run with ``Env0._run_test_solo``
```
import numpy as np
from Robots.ressources.solo import Solo
from Robots.ressources.plane import Plane
robot = Solo(class_terrain=Plane, GUI=True)
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
```

## Talos
This test can be run with ``Env0._run_test_talos``
```
import numpy as np
From Robots.ressources.talos import Talos
from Robots.ressources.plane import Plane
robot = Talos(Plane, GUI=True)
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
    robot.moveRobot(q_des, v_des, real_time=True, printInfos=True)
    i+=1
    print(i,"/",counter_reset)
    if i%counter_reset==0: 
        robot.reset()
        i=0
```

## Env0
This test can be run with ``Env0._run_test_env``.
```
import numpy as np
from Robots.envs.env0 import Env0
env = Env0(GUI=True)
action = np.array([0.]*Env0.action_dim) # The robot keeps all its joints straight
while True:
    obs, reward, done, _ = env.step( action )
    if done:
        input("Press to restart ...")
        env.reset()
```
