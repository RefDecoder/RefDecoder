config_name=$1
N_GPUS=$2

if [ -z "$config_name" ] || [ -z "$N_GPUS" ]; then
    echo "Usage: bash scripts/run_train.sh <config_name> <n_gpus>"
    exit 1
fi

yaml="configs/train/${config_name}.yaml"
exp_name="RefDecoder"

n_HOST=1
elastic=1
GPUName="H200_${N_GPUS}"

nvidia-smi

out_dir_name="${exp_name}-${n_HOST}nodes-e${elastic}-${GPUName}"
res_root="./debug"

mkdir -p $res_root/$out_dir_name

torchrun \
--nproc_per_node=${N_GPUS} --nnodes=1 --master_port=16700 \
train.py \
--base $yaml \
-t --devices ${N_GPUS} \
lightning.trainer.num_nodes=1 \
--name ${out_dir_name} \
--logdir $res_root \
--auto_resume true 2>&1 | tee $res_root/$out_dir_name/training.log