#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/adv_training_spbn_1/best.pth
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/spbn_best.pth --spbn
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/eval.json --restore ./results/train/nospbn_best.pth
#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/eval.json --restore ./results/train/clean/best.pth