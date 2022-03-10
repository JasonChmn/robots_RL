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

### Variables in config.py
- ``MODE_CONTROL`` : Select one of the control mode "PD" or "TORQUE". Torque is a bit slower to train compare to PD, use the one that works for your task.
- ``NB_MAX_STEPS`` : The number of steps maximum per episode during training (to tune).
- ``NAME_ROBOT`` : Select the robot you want to use "talos" or "solo".
- ``TRESHOLD_DEAD`` : lower and upper bound for root position Z of the robot. If not in these bounds, we end the episode (to tune, too low or too high?).
- ``Q_DESIRED_ONLY`` : With PD controller, if True, v_des is always equal to 0. if False, v_des is defined as an action.
- ``IS_POS_BASE_ONE_DIMENSION`` : If true, we keep only the Z value of base position (Always True here).

### Variables in talos.py and solo.py
- **[TO SET] ``PATH_URDF`` : Path to the URDF file of the robot (You need to set it for each robot).**
- ``HIGH_GAINS`` : Gains and Frequencies of the robots need to be tuned to be used with RL. 
               if False, we use the default gains and frequencies in Gepetto's experiment (used with feedforward couples).
               if True, we set different gains and frequencies for RL (Use this one and tune it).
- ``FREQUENCY_SOLO_HZ / FREQUENCY_TALOS_HZ``, ``GAINS_P``, ``GAINS_D`` : These values need to be tuned if the robot does not have enough strength (can not move) or his actions are jerky and unstable. You can test different combinations of gains and frequencies with the class test_PD.py. For SOLO, gains P and D for all joints are the same ``(GAINS_P_ALL/GAINS_D_ALL)``. For TALOS, gains are defined in function ``_getGainsPD()``.
- ``FREQUENCY_UPDATE_CONTROL_HZ`` : Frequency at which we update the control (PD or torque). Read below ``moveRobot()`` function to understand how it works.
- (For Solo) ``SOLO_NAME`` : Select solo8 or solo12. Solo12 has 4 more controlled joints on shoulders.
- ``MAX_TORQUES`` : Maximum torque than can be applied on one joint. Lower may result in smoother results (read in PAPERS section "Torque control in RL").
- ``DIVIDE_BOUNDS_TORQUES`` : Further reduce the torques bounds. For example if we set a ``MAX_TORQUES=2`` and ``DIVIDE_BOUNDS_TORQUES=4``, the maximum bound of a torque will be ``[ -2/4, 2/4 ] = [ -0.5, 0.5 ]``.
                                                 
### Functions
- ``reset()`` : Reset the robot position/orientation and forces applied on it. The robot's starting configuration is defined by parameters ``_robot_start_pos`` and ```_robot_start_orientation``.
- ``moveRobot()`` : Use the PD controller on the robot for q_des qnd v_des. 
                Each call of this function runs the PD controller for 1/FREQUENCY_UPDATE_CONTROL_HZ seconds.
                The PD controller computes the torques and apply it on the robot every ``1/FREQUENCY_TALOS_HZ`` seconds (or ``1/FREQUENCY_SOLO_HZ``).
                Example for ``FREQUENCY_UPDATE_CONTROL_HZ=2000`` and ``FREQUENCY_UPDATE_CONTROL_HZ=50`` :
                moveRobot() for ``1/50=0.02 sec`` with an update of torques every ``1/2000=0.0005 sec``.
- ``moveRobot_torques`` : use the torque controller on the robot.
-               Each call of this functions runs the torque controller for 1/FREQUENCY_UPDATE_CONTROL_HZ seconds.
-               It applies constant torques on the robot for all this duration.
-               The same torque is re-applied every ``1/FREQUENCY_TALOS_HZ`` (or ``1/FREQUENCY_SOLO_HZ``) seconds (I do not know if setting a high value has an impact?).
- ``getJointsState()``  : Return q_mes and v_mes of all joints.
- ``getJointsBounds()`` : Return bounds for q_mes and v_mes of all joints.
- ``getControlledJointsState()``  : Return q_mes and v_mes of controlled joints.
- ``getControlledJointsBounds()`` : Return bounds for q_mes and v_mes of controlled joints.
- ``getControlledJointsTorquesBounds()`` : Return bounds for torques of controlled joints.
- ``getBasePosOri()`` : Return base position and orientation.
- ``getBaseVel()``    : Return base linear and angular velocity.

## TESTS on both robots (in run_env.py)
```
# TEST - environment
Env._run_test_env()          # Move robot with null action (0.)
# TEST - on robot class
Env._run_test_robot()        # Move to default position.
Env._run_test_robot_reset()  # Check if the robot is reset correctly after each episode.
Env._run_test_robot_joints() # Check bounds for each controllable joint of the robot
```

## PLAY or TRAIN
To train the robot : set ``TRAIN = True``.

To play a trained policy : 
- set ``TRAIN = False``.
- model_name = PATH_TO_YOUR_MODEL (example : "logs/model__880000_steps")

## REINFORCEMENT LEARNING
We use PPO2 from stable baseline. You could also use another algorithm like td3 or sac.
In run_env.py, the network is a MlpPolicy: ``from stable_baselines.common.policies import MlpPolicy as CustomPolicy``).
MlpPolicy: https://stable-baselines.readthedocs.io/en/master/modules/policies.html is a 2x64 layers.
To define your own policy, you can comment the line above and uncomment ``class CustomPolicy(FeedForwardPolicy)``, https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html

All PPO2 hyperparameters are set in function ``def getModel(envVec)``, you have to tune it in function of your task. For hyperparameters tuning : https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

The reward is defined in both env_torque.py and env_PD.py, in ``getReward()`` function. The actual task is just to stay inside the bounds ``TRESHOLD_DEAD``.
For both control modes, this task is performed in ~1M steps with 8 parallel workers (in run_env.py, ``train()`` function).

## TO DO
- Design your own reward in function of the task to perform.
- Select the best control mode for your task (experiment and read papers below).
- Tune : Gains P/D, Frequencies, TRESHOLD_DEAD, NB_MAX_STEPS

## PAPERS
- Emergence of Locomotion Behaviours in Rich Environments : https://arxiv.org/abs/1707.02286 (video: https://www.youtube.com/watch?v=hx_bgoTF7bs&ab_channel=DeepMind)
- DeepLoco : https://www.cs.ubc.ca/~van/papers/2017-TOG-deepLoco/
- DeepGait : https://arxiv.org/abs/1909.08399
- Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter? : https://arxiv.org/abs/1611.01055
- Torque control in RL : https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0383251
