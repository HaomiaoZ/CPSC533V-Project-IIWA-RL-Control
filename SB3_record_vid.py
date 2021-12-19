# SB3 has the functionality, but just to be consitent with PD controller
import gym
import torch
import numpy as np
import cv2

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from iiwa_env_gym import IIWAEnvGym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

def record_vid(args):
    #torch.autograd.set_detect_anomaly(True)

    env= Monitor(IIWAEnvGym(target_type = args.target_type, render=True))
    video_frames =[]
    if args.policy == "PPO":
        model = PPO.load(args.model_path, env =env)
    elif args.policy =="SAC":
        model = SAC.load(args.model_path, env=env)

    obs = env.reset()
    for i in range(3):
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # rendering image
            frame = env.render(mode='rgb_array')
            video_frames.append(frame)

            if done:
                obs = env.reset()
                break

    out = cv2.VideoWriter('out_{}_{}.mp4'.format(args.policy,args.target_type),cv2.VideoWriter_fourcc('m','p','4','v'), 30, (1920,1080))
    for frame in video_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_type', type=str, default=None, help='[Point, Box, Box_pos,  Random]')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs/episodes to run')
    parser.add_argument('--policy',type =str,default = "PPO",help="[PPO, SAC]")
    parser.add_argument('--model_path',type=str,default=None,help="SB3 Model (.zip) you would like to record")

    args = parser.parse_args()

    record_vid(args)