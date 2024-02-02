#!/bin/bash
cd ..


for imb_ratio in 0.005 0.01 0.02 0.1; do
    CUDA_VISIBLE_DEVICES=0 python main_coop_prp_langevin_allclsloss.py \
    --workers 13 \
    --bz_trn 512 \
    --bz_tst 1024 \
    --dataset CIFAR10_LT \
    --cifar_imb_ratio ${imb_ratio} \
    --clip_config_path config/spec_nctx15.yaml \
    --num_classes 10 \
    --im_enc_type clip_rn50 \
    --epochs 200 \
    --burn_in_epoch 250 \
    --max_samples 20 \
    --infer_at_epoch 20 \
    --lr_prompt 0.05 \
    --lr_type multistep \
    --list_steplr 250 \
    --epoch_ckpt_save 40 \
    --tau 1 \
    --wd 1e-9 \
    --tensorboard log_cifar10/imb_${imb_ratio}/rn50_prp_spec/NoNoise_Base-ckpt/T_tau1_spec15_lr.05_wd1e-9
done