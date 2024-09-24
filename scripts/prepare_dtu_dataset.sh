# convert DTU dataset to colmap format
GPU_ids=3
# <your DTU_Rectified path>
original_dtu_path=./data/DTU/Rectified/
# <your DTU path>
output_path=./data/mvpgs_dataset_0920/dtu_colmap_dataset_0923/

scenes=(scan38 scan40 scan55 scan63 scan110 scan114 scan21 scan30 scan31 scan8 scan34 scan41 scan45 scan82 scan103)
for scene in "${scenes[@]}"
do
    CUDA_VISIBLE_DEVICES=$GPU_ids python dtu_colmap.py --data_dir $original_dtu_path/$scene --out_dir $output_path/$scene
done