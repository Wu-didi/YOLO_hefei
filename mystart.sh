export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --master_port=29502 --nproc_per_node=2 tools/train.py 