#CUDA_VISIBLE_DEVICES=1 python main.py --cfg_path configs/train.json --alg adv_training_nospbn --resume
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/train.json --alg adv_training_spbn --spbn --resume 