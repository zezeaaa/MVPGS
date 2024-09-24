GPU_ids=0
scenes=(scan38 scan40 scan55 scan63 scan110 scan114 scan21 scan30 scan31 scan8 scan34 scan41 scan45 scan82 scan103)
input_views=3
unseen_num=360
mvs_config=./mvs_modules/configs/config_mvsformer.json

# <your DTU path> (output path of scripts/prepare_dtu_dataset.sh)
data_path=./data/mvpgs_dataset_0920/dtu_colmap_dataset/
# <your DTU_mask path>
dtu_mask_path=./data/DTU/submission_data/idrmasks/

# downsample by 4
res=4
output_path=./output/DTU$input_views\_$res\x/

for scene in "${scenes[@]}"
do
    echo ========================= DTU Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene --dataset DTU \
    --stage train --input_views $input_views --iterations 10000 --densify_until_iter 5000 --total_virtual_num $unseen_num \
    --mono_depth_loss_w 0.01 --mvs_config $mvs_config --dtu_mask_path $dtu_mask_path

    echo ========================= DTU Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/$scene -m $output_path/$scene -r $res
    
    echo ========================= DTU Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= DTU Finish: $scene =========================
done

# downsample by 2
res=2
output_path=./output/DTU$input_views\_$res\x/

for scene in "${scenes[@]}"
do
    echo ========================= DTU Train: $scene ========================= 
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene --dataset DTU \
    --stage train --input_views $input_views --iterations 10000 --densify_until_iter 5000 --total_virtual_num $unseen_num \
    --mono_depth_loss_w 0.01 --mvs_config $mvs_config --dtu_mask_path $dtu_mask_path

    echo ========================= DTU Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -s $data_path/$scene -m $output_path/$scene -r $res
    
    echo ========================= DTU Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= DTU Finish: $scene =========================
done
