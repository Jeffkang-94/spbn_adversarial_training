CUDA_VISIBLE_DEVICES=0 python main.py --spbn\
            --cfg_path configs/eval.json \
            --restore ./results/train/adv_training_spbn_1/best.pth \
            --attack_steps 7 \
            --epsilon 8 \
            --attack PGD

#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/spbn_best.pth --spbn
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/clean/best.pth