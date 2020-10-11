
CUDA_VISIBLE_DEVICES=0,1 python -u -m torch.distributed.launch --nproc_per_node=2 train.py --dataset coco --coco_path /home1/datasets/coco/coco2017 --csv_classes cls_name.csv --epochs 10