GPU_ids=3
scenes=(Ballroom Barn Church Family Francis Horse Ignatius Museum)
res=1
input_views=3
unseen_num=360
mvs_config=./mvs_modules/configs/config_mvsformer.json

# <your T&T path>
data_path=./data/mvpgs_dataset_0920/Tanks/
output_path=./output/Tanks$input_views\_$res\x/

for scene in "${scenes[@]}"
do
    echo ========================= Tank Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene --dataset Tank \
    --stage train --input_views $input_views --iterations 20000 --densify_until_iter 10000 --total_virtual_num $unseen_num \
    --mvs_config $mvs_config

    echo ========================= Tank Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/$scene -m $output_path/$scene -r $res
    
    echo ========================= Tank Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= Tank Finish: $scene =========================
done