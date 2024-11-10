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

@hydra.main(config_name="dp3_realworld.yaml",
            version_base=None,
            config_path=str(pathlib.Path(__file__).parent.joinpath(
                'diffusion_policy_3d', 'config'))
)
def main(cfg: DictConfig):
    training_use_ema = False

    workspace = TrainDP3Workspace(cfg=cfg)

    # best_ckpt_path = workspace.get_checkpoint_path(tag="best")
    ckpt_dir = '/home/mei/workspace/DP3-Compliant-Gripper/3D-Diffusion-Policy/data/outputs/realworld_test-dp3_realworld-0001_seed0/checkpoints'
    best_ckpt_path = pathlib.Path(ckpt_dir).joinpath("epoch=0400-test_mean_score=-0.001.ckpt")
    if best_ckpt_path.is_file():
        print(f"Resuming from checkpoint {best_ckpt_path}")
        workspace.load_checkpoint(path=best_ckpt_path)

    # TODO: change runner settings
    env_runner = RealworldRunner(output_dir='./',
                                eval_episodes=5)
    # assert isinstance(env_runner, BaseRunner) # TODO: why not instance

    policy = workspace.model
    if training_use_ema:
        policy = workspace.ema_model
    policy.eval()
    policy.cuda()

    runner_log = env_runner.run(policy=policy, use_force=False)

    cprint(f"---------------- Eval Results --------------", 'magenta')
    for key, value in runner_log.items():
        if isinstance(value, float):
            cprint(f"{key}: {value:.4f}", 'magenta')

if __name__ == "__main__":
    main()
    