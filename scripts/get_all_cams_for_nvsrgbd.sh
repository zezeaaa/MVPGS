# <your NVS-RGBD path>
dataset_path=./data/mvpgs_dataset_0920/NVS-RGBD/
scenes=(scene01 scene02 scene03 scene04 scene05 scene06 scene07 scene08)

# zed2
for scene in "${scenes[@]}"
do
    data_path=$dataset_path/zed2/$scene/
    echo $data_path
    if [ "$scene" == "scene05" ]; then
        colmap mapper --database_path $data_path/database.db --image_path $data_path/images --output_path $data_path/sparse/ --Mapper.init_min_tri_angle 4
    else
        colmap mapper --database_path $data_path/database.db --image_path $data_path/images --output_path $data_path/sparse/
    fi

    colmap model_converter --input_path $data_path/sparse/0 --output_path $data_path/sparse/0 --output_type TXT
done

# kinect
for scene in "${scenes[@]}"
do
    data_path=$dataset_path/kinect/$scene/
    echo $data_path
    colmap mapper --database_path $data_path/database.db --image_path $data_path/images --output_path $data_path/sparse/
    colmap model_converter --input_path $data_path/sparse/0 --output_path $data_path/sparse/0 --output_type TXT
done