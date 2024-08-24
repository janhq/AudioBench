export AZURE_OPENAI_KEY=""
export OPENAI_API_KEY=""

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
MODEL_PATH_OR_ID="/home/root/BachVD/model_zoo/llama3.1-s-instruct-2024-08-19-epoch-3/"
BATCH_SIZE=1
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
export CUDA_VISIBLE_DEVICES=0
# SI
DATASET=openhermes_audio_test
METRICS=gpt4_judge
python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_path_or_id $MODEL_PATH_OR_ID \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES

DATASET=alpaca_audio_test
METRICS=gpt4_judge
python src/main_evaluate.py \
    --dataset_name $DATASET \
    --model_path_or_id $MODEL_PATH_OR_ID \
    --batch_size $BATCH_SIZE \
    --overwrite $OVERWRITE \
    --metrics $METRICS \
    --number_of_samples $NUMBER_OF_SAMPLES

