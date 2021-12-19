from stable_baselines3.common.env_checker import check_env
from iiwa_env_gym import IIWAEnvGym

# import your env and check it
env = IIWAEnvGym("Box")
# It will check your custom environment and output additional warnings if needed
check_env(env)