export AZURE_OPENAI_KEY=""
export OPENAI_API_KEY=""
DATASET=$1
MODEL=$2
BATCH_SIZE=$3
OVERWRITE=$4
METRICS=$5
NUMBER_OF_SAMPLES=$6


export CUDA_VISIBLE_DEVICES=2,3


python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES
