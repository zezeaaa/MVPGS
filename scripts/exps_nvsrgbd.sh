GPU_ids=2
scenes=(scene01 scene02 scene03 scene04 scene05 scene06 scene07 scene08)
unseen_num=360
res=2
mvs_config=./mvs_modules/configs/config_mvsformer.json

# <your NVS-RGBD path>
data_path=./data/mvpgs_dataset_0920/NVS-RGBD/

# zed branch
output_path=./output/NVSRGBD/zed3_$res\x/
for scene in "${scenes[@]}"
do
    echo ========================= NVSRGBD zed Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/zed2/$scene -r $res -m $output_path/$scene --dataset NVSRGBD \
    --stage train --iterations 20000 --densify_until_iter 10000 --total_virtual_num $unseen_num \
    --mvs_config $mvs_config

    echo ========================= NVSRGBD zed Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/zed2/$scene -m $output_path/$scene -r $res

    echo ========================= NVSRGBD zed Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= NVSRGBD zed Finish: $scene =========================
done

# kinect brance
output_path=./output/NVSRGBD/kinect3_$res\x/
for scene in "${scenes[@]}"
do
    echo ========================= NVSRGBD zed Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/kinect/$scene -r $res -m $output_path/$scene --dataset NVSRGBD \
    --stage train --iterations 20000 --densify_until_iter 10000 --total_virtual_num $unseen_num \
    --mvs_config $mvs_config

    echo ========================= NVSRGBD zed Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/kinect/$scene -m $output_path/$scene -r $res

    echo ========================= NVSRGBD zed Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= NVSRGBD zed Finish: $scene =========================
done