import gym
import torch

from iiwa_env import IIWAEnv
from iiwa_env_gym import IIWAEnvGym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(policy, target_type, env='IIWA_Position', num_test_episodes=10, render=False, verbose=False):
    test_env = IIWAEnvGym(render=render,target_type=target_type)
    test_rewards = []
    for i in range(num_test_episodes):
        state = test_env.reset()
        episode_total_reward = 0
        while True:
            state = torch.tensor([state], device=device, dtype=torch.float32)
            action, _, _ = policy.step(torch.as_tensor(state, dtype=torch.float32).to(device))
            next_state, reward, done,_= test_env.step(action.flatten())
            
            '''
            if render:
                test_env.render(mode='human')
            '''

            episode_total_reward += reward
            state = next_state
            if done:
                if verbose:
                    print('[Episode {:4d}/{}] [reward {:.1f}]'
                        .format(i, num_test_episodes, episode_total_reward))
                break
        test_rewards.append(episode_total_reward)
    test_env.close()
    return sum(test_rewards)/num_test_episodes


if __name__ == "__main__":
    import argparse
    from models import ActorCritic

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=None, type=str,
        help='Path to the model weights.')
    parser.add_argument('--env', default=None, type=str,
        help='Name of the environment.')
    
    parser.add_argument('--target_type', type=str, default=None, help='[Point, Cube, Random]')
    parser.add_argument('--num_test_episodes', type=int, default=10, help='number of test episode')
    args = parser.parse_args()
    env = IIWAEnv(target_type=args.target_type)
    model = ActorCritic(21, 7, False).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    env.close()

    eval_policy(policy=model, env=args.env, render=True, target_type = args.target_type, verbose=True,num_test_episodes=args.num_test_episodes )
