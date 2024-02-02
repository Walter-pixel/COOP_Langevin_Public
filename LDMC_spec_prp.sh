#!/bin/bash


source /home/walterl/ml20_scratch/walterl/miniconda/bin/activate torch_env
cd /home/walterl/ml20_scratch/walterl/Langevin_Dynamics/COOP_Langevin



## Below is to train the baseckpt without any noise
#python3 main_coop_prp_langevin_allclsloss.py \
#--workers 26 \
#--bz_trn 512 \
#--bz_tst 1024 \
#--dataset CIFAR10_LT \
#--cifar_imb_ratio 0.005 \
#--clip_config_path config/spec_nctx15.yaml \
#--num_classes 10 \
#--im_enc_type clip_rn50 \
#--epochs 200 \
#--burn_in_epoch 250 \
#--max_samples 20 \
#--infer_at_epoch 20 \
#--lr_prompt 0.05 \
#--lr_type multistep \
#--list_steplr 250 \
#--epoch_ckpt_save 40 \
#--tau 1 \
#--wd 1e-9 \
#--tensorboard log_cifar10/imb200/rn50_prp_spec/NoNoise_Base-ckpt/T_tau1_spec15_lr.05_wd1e-9




python3 main_coop_prp_langevin_allclsloss.py \
--ckpt_path log_cifar10/imb200/rn50_prp_spec/NoNoise_Base-ckpt/T_tau1_spec15_lr.05_wd1e-9/saved_prompts_at_training/prp_epoch_120.pth \
--workers 26 \
--bz_trn 512 \
--bz_tst 1024 \
--dataset CIFAR10_LT \
--cifar_imb_ratio 0.005 \
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
--tensorboard log_cifar10/imb200/rn50_prp_spec/keeptrain_with_mynoise/T_tau1_spec15_lr.05_wd1e-9/tau1_stplr5e-2.140.200_brn200_max80




