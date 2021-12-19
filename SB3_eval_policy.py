import gym
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from iiwa_env_gym import IIWAEnvGym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def SB3_eval_policy(args):
    #torch.autograd.set_detect_anomaly(True)

    env= Monitor(IIWAEnvGym(target_type = args.target_type, render =args.render))

    if args.model_path != None:
        if args.policy == "PPO":
            model = PPO.load(args.model_path, env =env)
        elif args.policy =="SAC":
            model = SAC.load(args.model_path, env=env)
    else:
        raise Exception("model path is not provided")
    
    rewards = evaluate_policy(model,env,n_eval_episodes=args.num_test_episodes,return_episode_rewards=True)
    env.close()
    print('Average Reward of {:d} episode is {:.1f}, STD is: {:.1f}'.format(args.num_test_episodes, sum(rewards[0])/len(rewards[0]), np.std(rewards[0])))
    print('Average Episode Length is: {:.1f}, STD is: {:.1f}'.format(sum(rewards[1])/len(rewards[1]), np.std(rewards[1])))
    # return reward and episode length in [0] and [1]
    return rewards



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_type', type=str, default=None, help='[Point, Box, Random]')
    parser.add_argument('--policy',type =str,default = "PPO",help="[PPO, SAC]")
    parser.add_argument('--model_path',type=str,default=None,help="SB3 Model (.zip) you would like to keep training")
    parser.add_argument('--num_test_episodes', type=int, default=10, help='number of test episode')
    parser.add_argument('--render',type=bool,default=True,help="SB3 Model (.zip) you would like to keep training")

    args = parser.parse_args()

    SB3_eval_policy(args)