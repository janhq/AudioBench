

DATASET=$1
MODEL=$2
BATCH_SIZE=$3
OVERWRITE=$4
METRICS=$5
NUMBER_OF_SAMPLES=$6


export CUDA_VISIBLE_DEVICES=0,1,2,3


python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES
