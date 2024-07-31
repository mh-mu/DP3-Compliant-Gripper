import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from diffusion_policy_3d.env import MetaWorldEnv
from gymnasium.wrappers import EnvCompatibility

def main(args):
    env_name = args.env_name

    save_dir = os.path.join(args.model_save_dir, 'metaworld_'+args.env_name+'_RL_model')
    
    # if os.path.exists(save_dir):
    #     cprint('RL model already exists at {}'.format(save_dir), 'red')
    #     cprint("If you want to overwrite, delete the existing directory first.", "red")
    #     cprint("Do you want to overwrite? (y/n)", "red")
    #     user_input = 'y'
    #     if user_input == 'y':
    #         cprint('Overwriting {}'.format(save_dir), 'red')
    #         os.system('rm -rf {}'.format(save_dir))
    #     else:
    #         cprint('Exiting', 'red')
    #         return
    # os.makedirs(save_dir, exist_ok=True)

    env = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)
    env = EnvCompatibility(env, 'none')
    check_env(env)

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=args.num_timesteps)
    model.save(args.model_save_dir)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # eval
    obs = env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hammer')
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--model_save_dir', type=str, default="/data/outputs_rl/" )

    args = parser.parse_args()
    main(args)