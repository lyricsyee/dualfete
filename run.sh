#!/bin/zsh

dataset=LA
datapath=data/LA

gpu=0
cutmix_prob=0.5
threshold=0.7
consist_ramp=20
faster=5
labeled_num=8

exp_name=exp_dualfete
python code/train_dualfete.py --dataset_name $dataset --root_path $datapath --exp l8_dualfete/${exp_name} --model vnet --patch_size 112 112 80 --num_classes 2 --labeled_num $labeled_num --gpu $gpu \
                              --flag_rotflip 1 --step_normgrad 1 --consistency_rampup $consist_ramp --softpl_mask_thd $threshold --cutmix_prob $cutmix_prob --faster_factor $faster 
python code/test_performance.py --dataset $dataset --root_path $datapath --exp l8_dualfete/${exp_name} --model vnet --gpu $gpu --labeled_num $labeled_num 
