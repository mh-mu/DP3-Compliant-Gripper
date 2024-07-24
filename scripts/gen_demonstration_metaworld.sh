# bash scripts/gen_demonstration_metaworld.sh basketball



cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 1 \
            --root_dir "../../../../../implicit_force_simulation/data/" \
            #--root_dir "../../3D-Diffusion-Policy/data/" \
