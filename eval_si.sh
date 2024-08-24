MODEL=$1
BATCH_SIZE=1
DATASET1=openhermes_audio_test
DATASET2=alpaca_audio_test
METRICS=gpt4_judge
OVERWRITE=True
NUMBER_OF_SAMPLES=1

mkdir -p log

export CUDA_VISIBLE_DEVICES=0


python src/main_evaluate.py \
    --dataset_name $DATASET1 \
    --model_path_or_id $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES

python src/main_evaluate.py \
    --dataset_name $DATASET2 \
    --model_path_or_id $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES
