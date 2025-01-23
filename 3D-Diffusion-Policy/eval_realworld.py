import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffusion_policy_3d', 'env_runner')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffusion_policy_3d', 'policy')))

from realworld_runner import RealworldRunner
from base_runner import BaseRunner
from dp3 import DP3Realworld
from train import TrainDP3Workspace
from cprint import *
import hydra
import pathlib
from omegaconf import DictConfig
from icecream import ic

@hydra.main(config_name="dp3_realworld.yaml",
            version_base=None,
            config_path=str(pathlib.Path(__file__).parent.joinpath(
                'diffusion_policy_3d', 'config'))
)
def main(cfg: DictConfig):
    ic()
    ic(cfg['task_name']) # TODO: find out where to input task yaml during rollout
    training_use_ema = False

    workspace = TrainDP3Workspace(cfg=cfg)

    # best_ckpt_path = workspace.get_checkpoint_path(tag="best")
    # ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_circle_30Hz-dp3_realworld-1030_seed0/checkpoints'
    # ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_touch_10Hz-dp3_realworld-0002_seed0/checkpoints'
    # ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_contact_10Hz-dp3_realworld-0003_seed0/checkpoints'
    ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_contact_compliant_10Hz-dp3_realworld-0002_seed0/checkpoints'
    best_ckpt_path = pathlib.Path(ckpt_dir).joinpath("latest.ckpt")
    if best_ckpt_path.is_file():
        print(f"Resuming from checkpoint {best_ckpt_path}")
        workspace.load_checkpoint(path=best_ckpt_path)

    env_runner = RealworldRunner(output_dir='./',
                                eval_episodes=10,
                                max_steps=300,
                                # fps=30, # 30Hz
                                fps=10, # 10Hz
                                n_obs_steps=2,
                                n_action_steps=4,)
    # assert isinstance(env_runner, BaseRunner) # TODO: why not instance

    policy = workspace.model
    if training_use_ema:
        policy = workspace.ema_model
    policy.eval()
    policy.cuda()

    runner_log = env_runner.run(policy=policy, use_force=cfg['policy']['use_force'])

    cprint(f"---------------- Eval Results --------------", 'magenta')
    for key, value in runner_log.items():
        if isinstance(value, float):
            cprint(f"{key}: {value:.4f}", 'magenta')

if __name__ == "__main__":
    main()
    