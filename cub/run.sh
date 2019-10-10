#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0

cd ..

MODEL=base
DATA=cub
BACKBONE=resnet101
SAVE_PATH=./${DATA}/checkpoints/${MODEL}

mkdir -p ${SAVE_PATH}

python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 192 --lr 0.001 --epochs 90 --is_fix --pretrained &> ${SAVE_PATH}/fix.log
python main.py -a ${MODEL} -d ${DATA} -s ${SAVE_PATH} --backbone ${BACKBONE} -b 16 --lr 0.0001 --epochs 90 --resume ${SAVE_PATH}/fix.model &> ${SAVE_PATH}/ft.log

