export CUDA_VISIBLE_DEVICES=0,1

# cd ..
python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=25692 train_distributed.py \
-c configs/pa_po_nuscenes_trainval.yaml \
-l /visionai-postech/eccv24/nusc_logs \
-w nusc_final \
-n nusc_tmp \

#-n nusc_late_cont_head_attnfea \
#-x \
