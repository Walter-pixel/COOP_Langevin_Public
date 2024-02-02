#!/bin/bash
cd ..


for imb_ratio in 0.005 0.01 0.02 0.1; do

    CUDA_VISIBLE_DEVICES=0  python3 main_coop_prp_langevin_allclsloss.py \
    --ckpt_path log_cifar10/imb_${imb_ratio}/rn50_prp_spec/NoNoise_Base-ckpt/T_tau1_spec15_lr.05_wd1e-9/saved_prompts_at_training/prp_epoch_160.pth \
    --workers 26 \
    --bz_trn 512 \
    --bz_tst 1024 \
    --dataset CIFAR10_LT \
    --cifar_imb_ratio ${imb_ratio} \
    --clip_config_path config/spec_nctx15.yaml \
    --num_classes 10 \
    --im_enc_type clip_rn50 \
    --epochs 400 \
    --burn_in_epoch 200 \
    --max_samples 80 \
    --infer_at_epoch 40 \
    --lr_prompt 5e-2 \
    --lr_type multistep \
    --list_steplr 140 200 \
    --epoch_ckpt_save 1000 \
    --tau 1 \
    --wd 1e-5 \
    --tensorboard log_cifar10/imb_${imb_ratio}/rn50_prp_spec/keeptrain_LD/T_tau1_spec15_lr.05_wd1e-9/ckpt160_tau1_stplr5e-2.140.200_brn200_max80
done