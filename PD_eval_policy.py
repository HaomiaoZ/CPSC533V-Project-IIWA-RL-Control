# evaluate the PD position controller as a baseline

import pybullet as p
import time
import pybullet_data
import numpy as np
from pybullet_utils import bullet_client as bc

import gym
import torch

from iiwa_env import IIWAEnv
from iiwa_env_gym import IIWAEnvGym

def PDController(IIWAEnvGym,observation):
    physicsClient =IIWAEnvGym.getPhysicsClient()
    IIWAId = IIWAEnvGym.getIIWAId()
    target_eef_pos = observation[14:17]
    target_eef_ori = observation[17:]
    target_eef_pos[2] = target_eef_pos[2] *0.9+0.6
    action = physicsClient.calculateInverseKinematics(IIWAId,6,target_eef_pos,target_eef_ori,\
    [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],\
        [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],\
            [5.8, 4, 5.8, 4, 5.8, 4, 6],\
            [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0])
    return np.divide(action,IIWAEnvGym.getJointPositionLimit())

def PD_eval_policy(env, render, target_type,  num_test_episodes,verbose=True ):
    test_env = IIWAEnvGym(render=render,target_type=target_type)
    test_rewards = []
    test_episodes_length =[]
    for i in range(num_test_episodes):
        state = test_env.reset()
        episode_total_reward = 0
        episode_length = 0
        while True:
            action = PDController(test_env,state)
            next_state, reward, done,_= test_env.step(np.array(action).flatten())
            
            '''
            if render:
                test_env.render(mode='human')
            '''

            episode_total_reward += reward
            episode_length+=1
            state = next_state
            if done:
                if verbose:
                    print('[Episode {:4d}/{}] [reward {:.1f}]'
                        .format(i, num_test_episodes, episode_total_reward))
                break
        test_rewards.append(episode_total_reward)
        test_episodes_length.append(episode_length)
    test_env.close()
    print('Average Reward of {:d} episode is {:.1f}'.format(args.num_test_episodes, sum(test_rewards)/len(test_rewards)))
    print('Average Episode Length is: {:.1f}'.format(sum(test_episodes_length)/len(test_episodes_length)))
    return sum(test_rewards)/num_test_episodes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None, type=str,
        help='Path to the model weights.')
    parser.add_argument('--env', default=None, type=str,
        help='Name of the environment.')
    
    parser.add_argument('--target_type', type=str, default=None, help='[Point, Cube, Random]')
    parser.add_argument('--num_test_episodes', type=int, default=10, help='number of test episode')
    parser.add_argument('--render',type=bool,default=True,help="whether you would like to see graphics")

    args = parser.parse_args()

    PD_eval_policy(env=args.env, render=args.render, target_type = args.target_type, num_test_episodes=args.num_test_episodes,verbose=True )
