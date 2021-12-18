
import gym
import torch
import os

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from iiwa_env_gym import IIWAEnvGym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback,EvalCallback

def main(args):
    #torch.autograd.set_detect_anomaly(True)

    env= Monitor(IIWAEnvGym(target_type = args.target_type))

    #eval_callback will save the best model
    eval_env = Monitor(IIWAEnvGym(target_type = args.target_type))
    save_path = os.path.join('./logs/',args.policy,args.target_type)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path,
                             log_path=save_path, eval_freq=50*300,
                             deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=max(100,int(args.epochs/100))*300, save_path=os.path.join('./logs/CheckPoint/',args.policy,args.target_type))
    callback = CallbackList([checkpoint_callback, eval_callback])
    if args.keep_training_model_path != None:
        if args.policy == "PPO":
            model = PPO.load(args.keep_training_model_path, env =env)
        elif args.policy =="SAC":
            model = SAC.load(args.keep_training_model_path, env=env)

    elif args.policy =="PPO":
        policy_kwargs =dict(activation_fn=torch.nn.modules.activation.ReLU, net_arch=[dict(vf=[256,256],pi =[256,256])] )
        model = PPO('MlpPolicy', env, learning_rate =3e-4,gae_lambda=0.97, n_steps = 300, batch_size =300, \
            create_eval_env=True, policy_kwargs=policy_kwargs,verbose=1,tensorboard_log=save_path)
    
    elif args.policy =="SAC":
        policy_kwargs =dict(activation_fn=torch.nn.modules.activation.ReLU, net_arch=dict(qf=[256,256],pi =[256,256]))
        model = SAC('MlpPolicy', env, learning_rate =3e-4,create_eval_env=True,policy_kwargs=policy_kwargs,verbose=1,tensorboard_log=save_path)
    
    model.learn(total_timesteps=args.epochs*300, log_interval=100, callback=callback)

    #save a model at the end of training
    model.save("SB3_{}_IIWA_{}".format(args.policy,args.target_type))

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_type', type=str, default=None, help='[Point, Box, Random]')
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs/episodes to run')
    parser.add_argument('--policy',type =str,default = "PPO",help="[PPO, SAC]")
    parser.add_argument('--keep_training_model_path',type=str,default=None,help="SB3 Model (.zip) you would like to keep training")

    args = parser.parse_args()

    main(args)