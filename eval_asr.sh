MODEL=$1
BATCH_SIZE=1
OVERWRITE=True
METRICS=wer
DATASET=librispeech_test_clean
NUMBER_OF_SAMPLES=1

mkdir -p log

export CUDA_VISIBLE_DEVICES=0


python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_path_or_id $MODEL \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES

