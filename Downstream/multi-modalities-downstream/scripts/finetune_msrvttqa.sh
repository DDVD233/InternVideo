NUM_NODES=1
NUM_GPUS=4
JOB_NAME="vtc_msrvttqa_lr1e5"
MODEL_PREFIX="."
OUTPUT_DIR="/home/data/models/int_moma"
LOG_FILE="${OUTPUT_DIR}/logs/${JOB_NAME}/log.txt"
N_TASKS=`echo "${NUM_NODES}*${NUM_GPUS}" | bc`

python run.py with \
data_root=./meta_data \
num_gpus=4 \
num_nodes=1 \
per_gpu_batchsize=16 \
clip_finetune_msrvttqa \
num_frames=8 \
num_workers=8 \
batch_size=512 \
max_epoch=20 \
model_dir=/home/data/models/int_moma/models/vtc_msrvttqa_lr1e5 \
log_dir=/home/data/models/int_moma/result/vtc_msrvttqa_lr1e5 \
resume_from=/home/data/models/internvideo/msrvtt.ckpt \
save_checkpoints_interval=1000 \
learning_rate=1e-5 \
clip_qa_type=vtc_cap \
save_last=False \
save_top_k=0 \
max_steps=-1 \
clip=/home/data/models/internvideo/ViT-L-14.pt \
clip_type=kc_new \
load_path=/home/data/models/internvideo/msrvtt.ckpt

while [ ! -f ${LOG_FILE} ] ; do sleep 1; done
tail -f ${LOG_FILE}