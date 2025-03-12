import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffusion_policy_3d', 'env_runner')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'diffusion_policy_3d', 'policy')))

from realworld_runner import RealworldRunner
from dummy_runner import DummyRunner
from base_runner import BaseRunner
from dp3 import DP3Realworld
from train import TrainDP3Workspace
from cprint import cprint
import hydra
import pathlib
from omegaconf import DictConfig
from icecream import ic

# @hydra.main(config_name="dp3_realworld.yaml",
#             version_base=None,
#             config_path=str(pathlib.Path(__file__).parent.joinpath(
#                 'diffusion_policy_3d', 'config'))
# )

@hydra.main(config_name="dp3_realworld_dummy.yaml",
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
    # ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_contact_29_5Hz-dp3_realworld_horizon1-rate_new_seed6/checkpoints'
    ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_dummy-dp3_realworld_dummy-with_force_seed0/checkpoints'
    # ckpt_dir = '/home/mh2595/workspace/implicit_force_simulation/third_party/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/realworld_contact_10Hz-dp3_realworld-0003_seed0/checkpoints'
    best_ckpt_path = pathlib.Path(ckpt_dir).joinpath("epoch=0100-test_mean_score=-0.000.ckpt")
    if best_ckpt_path.is_file():
        print(f"Resuming from checkpoint {best_ckpt_path}")
        workspace.load_checkpoint(path=best_ckpt_path)

    # env_runner = RealworldRunner(output_dir='./',
    #                             eval_episodes=10,
    #                             max_steps=600,
    #                             fps=30, # 30Hz
    #                             # fps=10, # 10Hz
    #                             # fps=5, # 5Hz
    #                             n_obs_steps=2,
    #                             n_action_steps=10,) # TODO: check if action steps is correct
    
    env_runner = DummyRunner(output_dir='./',
                                eval_episodes=10,
                                max_steps=600,
                                fps=30, # 30Hz
                                n_obs_steps=2,
                                n_action_steps=8,
                                task_name='dummy_withforce_epoch100') # TODO: check if action steps is correct
    
    # assert isinstance(env_runner, BaseRunner) # TODO: why not instance

    policy = workspace.model
    if training_use_ema:
        policy = workspace.ema_model
    policy.eval()
    policy.cuda()

    runner_log = env_runner.run(policy=policy, save_video=True, use_force=cfg['policy']['use_force'])

    print("---------------- Eval Results --------------")
    for key, value in runner_log.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
    