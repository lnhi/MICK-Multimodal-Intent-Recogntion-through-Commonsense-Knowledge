#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 8 #4 6 10 12
    do
        for weight_fuse_relation in 0.5 #0.5 0.7
        do
            for weight_fuse_visual_comet in 0.4
            do
                python run.py \
                --dataset $dataset \
                --logger_name 'mick' \
                --method 'mick' \
                --data_mode 'binary-class' \
                --train \
                --save_results \
                --seed $seed \
                --gpu_id '0' \
                --video_feats_path 'video_feats.pkl' \
                --audio_feats_path 'audio_feats.pkl' \
                --text_backbone 'bert-base-uncased' \
                --config_file_name 'mick' \
                --results_file_name 'mick.csv' \
                --weight_fuse_relation $weight_fuse_relation \
                --weight_fuse_visual_comet $weight_fuse_visual_comet
            done
        done
    done
done