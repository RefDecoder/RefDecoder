config_name=$1
data_root=$2
out_root=$3
chunk_size=${4:-17}
height=${5:-480}
width=${6:-832}
device=${7:-cuda:0}

yaml="configs/inference/${config_name}.yaml"

if [ -z "$config_name" ] || [ -z "$data_root" ] || [ -z "$out_root" ]; then
    echo "Usage: bash scripts/run_inference.sh <config_name> <data_root> <out_root> [chunk_size] [height] [width] [device]"
    exit 1
fi

nvidia-smi

mkdir -p "$out_root"

python inference_video.py \
    --config_path "$yaml" \
    --data_root "$data_root" \
    --out_root "$out_root" \
    --chunk_size "$chunk_size" \
    --resolution "$height" "$width" \
    --device "$device" 2>&1 | tee "$out_root/inference.log"
