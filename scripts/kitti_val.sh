export CUDA_VISIBLE_DEVICES=0,1

# cd ..
python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=25692 val.py \
-c configs/pa_po_kitti_val.yaml \
-l kitti_logs/val.log
