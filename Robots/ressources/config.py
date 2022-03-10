
class Config:
	# ========= Type of control =========	
	CONTROL_LIST = ["PD", "TORQUE"]
	MODE_CONTROL = CONTROL_LIST[0] # PD
	#MODE_CONTROL = CONTROL_LIST[1] # Torque
	if MODE_CONTROL=="PD":
	    ENV_ROBOT_ID = 'env_robot-v0' # Name in Robots/__init__.py
	    NB_MAX_STEPS = 300 # To tune, 300 steps should be more than enough.
	elif MODE_CONTROL=="TORQUE":
	    ENV_ROBOT_ID = 'env_robot-v1' # Name in Robots/__init__.py
	    NB_MAX_STEPS = 300 # To tune

	# ========= Robot selection =========
	ROBOT_LIST = ["talos","solo"]
	#NAME_ROBOT = ROBOT_LIST[0] # talos
	NAME_ROBOT = ROBOT_LIST[1] # solo

	# ========= Height of robot and death treshold ========= 
	if NAME_ROBOT=="talos":
	    HEIGHT_ROOT = 1.0   # Height of the robot when standing
	    TRESHOLD_DEAD = [HEIGHT_ROOT-0.35, 2.0]  # Episode is over if the robot goes lower or higher
	elif NAME_ROBOT=="solo":
	    HEIGHT_ROOT = 0.23  # Height of the robot when standing
	    TRESHOLD_DEAD = [HEIGHT_ROOT-0.10, 0.55] # Episode is over if the robot goes lower or higher
	else:
	    input("Error, name of the robot not defined ...")

	# ========= Other ========= 
	Q_DESIRED_ONLY = True # [PD] True: v_des is null / False: v_des is defined in actions
	IS_POS_BASE_ONE_DIMENSION = True # If true, we keep only the Z value of base position (Always True here)