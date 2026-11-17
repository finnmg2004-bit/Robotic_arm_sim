import sys
if sys.prefix == '/home/finn-gorman/Projects/RoboArmSim/.pixi/envs/default':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/finn-gorman/Projects/RoboArmSim/install/arm_policy'
