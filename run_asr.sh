# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_PATH_OR_ID="/home/root/BachVD/model_zoo/llama3.1-s-instruct-2024-08-19-epoch-3/"
BATCH_SIZE=1
OVERWRITE=True
NUMBER_OF_SAMPLES=1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
export CUDA_VISIBLE_DEVICES=0
# SI
DATASET=librispeech_test_clean
METRICS=wer
python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_path_or_id $MODEL_PATH_OR_ID \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES


