import numpy as np
import sys, os
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

current_dir = os.path.dirname(os.path.abspath(__file__))
manlearn_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, 'manipulator_learning', 'manipulator_learning'))
print(manlearn_dir)
sys.path.append(manlearn_dir)

import manipulator_learning.sim.envs as manlearn_envs

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode reward and length
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.locals['rewards'][0])
            self.episode_lengths.append(self.locals['infos'][0]['episode']['l'])
            wandb.log({
                "episode_reward": self.episode_rewards[-1],
                "episode_length": self.episode_lengths[-1],
                "mean_100ep_reward": np.mean(self.episode_rewards[-100:]),
                "mean_100ep_length": np.mean(self.episode_lengths[-100:])
            })
        return True

# Initialize wandb
wandb.init(project="sim_force_pybullet", name="ppo-run", config={
    "algorithm": "PPO",
    "environment": "ThingPickAndInsertSucDoneImage"
})

# Create and wrap the environment
env = getattr(manlearn_envs, 'ThingPickAndInsertSucDoneImage')(state_data = ('pos','contact_force'))
env = EnvCompatibility(env, 'none')
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Initialize the PPO agent
model = PPO("MultiInputPolicy", env, verbose=1)

# Create the callbacks
wandb_callback = WandbCallback()
custom_callback = CustomWandbCallback()

# Train the agent
total_timesteps = 100000
model.learn(
    total_timesteps=total_timesteps,
    callback=[wandb_callback, custom_callback]
)

# Save the final model
model.save("ThingPickAndInsertSucDoneImage")

# Log the final model to wandb
wandb.save("ThingPickAndInsertSucDoneImage.zip")

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
wandb.finish()