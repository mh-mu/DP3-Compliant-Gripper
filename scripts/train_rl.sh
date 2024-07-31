# bash scripts/train_rl.sh basketball



task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python train_rl.py --env_name=${task_name} \
            --num_timesteps 2500 \
            # --model_save_dir "../../../../../implicit_force_simulation/data/" \
            
