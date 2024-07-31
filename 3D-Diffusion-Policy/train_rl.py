import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

from diffusion_policy_3d.env import MetaWorldEnv
from gymnasium.wrappers import EnvCompatibility
from cprint import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#example: 
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Example for a 3-channel image (adjust for your input shape)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

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

    # custom extractor
    extractor = None #TODO, example at the top
    out_feature_dim = 256
    policy_kwargs = dict(
        features_extractor_class=extractor,
        features_extractor_kwargs=dict(features_dim=out_feature_dim),
    )


    env = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)
    env = EnvCompatibility(env, 'none')
    check_env(env)
    cprint.info('Env check passed')

    model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.num_timesteps)
    model.save(args.model_save_dir)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # eval
    obs = env.reset()[0]
    for _ in range(1000):
        action, _states = model.predict(obs)
        #obs, rewards, dones, info = env.step(action) #this is gym format
        obs, reward, terminated, truncated, info = env.step(action) #this is gymnasium
        env.render()
        if terminated:
            obs = env.reset()[0]

    env.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='hammer')
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--model_save_dir', type=str, default="./data/outputs_rl/" )

    args = parser.parse_args()
    main(args)