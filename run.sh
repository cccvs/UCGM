#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/run_train_ipa.sh configs/tuning_few_steps/in1k256_sd15_klae_ddim.yaml