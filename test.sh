CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/eval.json --restore ./results/train/adv_training_spbn_2/best.pth --spbn --attack_steps 40
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/spbn_best.pth --spbn
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/clean/best.pth