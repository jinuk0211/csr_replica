sample.py
```shell
#!/bin/bash
source ~/miniconda3/bin/activate your_env

NUM_PROCESSES=4
NUM_MACHINES=1
GPU_IDS=0,1,2,3
MODEL_PATH="llava-hf/llava-1.5-7b-hf"
BASE_HF_MODEL_PATH="llava-hf/llava-1.5-7b-hf"
IS_HF="1"
DATASET_PATH="./data/CSR-Prompt-Dataset-12k.json"
IMAGES_DIR="./data/images/train2014"
OUTPUT_DIR="./outputs/sample"
DIVERSITY_PENALTY="3.0"
NUM_BEAMS="5"
NUM_BEAM_GROUP="5"
NUM_TOKEN_BEAMS="5"
MAX_LENGTH="1024"
MAX_NEW_TOKENS="70"
PERIOD_ID="29889"
WEIGHT_MAPPING_PATH="./model_mapping/key_mapping_hf_7b.json"

accelerate launch \
  --num_processes=$NUM_PROCESSES \
  --num_machines=$NUM_MACHINES \
  --gpu_ids=$GPU_IDS \
  ./sample.py \
  --model_path $MODEL_PATH \
  --base_hf_model_path $BASE_HF_MODEL_PATH \
  --is_hf $IS_HF \
  --dataset_path $DATASET_PATH \
  --images_dir $IMAGES_DIR \
  --output_dir $OUTPUT_DIR \
  --diversity_penalty $DIVERSITY_PENALTY \
  --num_beams $NUM_BEAMS \
  --num_beam_group $NUM_BEAM_GROUP \
  --num_token_beams $NUM_TOKEN_BEAMS \
  --max_length $MAX_LENGTH \
  --max_new_tokens $MAX_NEW_TOKENS \
  --period_id $PERIOD_ID \
  --weight_mapping_path $WEIGHT_MAPPING_PATH
```
score.py
```shell
#!/bin/bash
source ~/miniconda3/bin/activate your_env

NUM_PROCESSES=4
NUM_MACHINES=1
GPU_IDS=0,1,2,3
FOLDER_PATH="./outputs/sample"
OUTPUT_DIR="./outputs/score"
DATA_JSON="./data/CSR-Prompt-Dataset-12k.json"
IMAGE_DIR="./data/images/train2014"
CLIP_MODEL_PATH="openai/clip-vit-large-patch14-336"

accelerate launch \
  --num_processes=$NUM_PROCESSES \
  --num_machines=$NUM_MACHINES \
  --gpu_ids=$GPU_IDS \
  ./score.py \
  --folder_path $FOLDER_PATH \
  --output_dir $OUTPUT_DIR \
  --data_json $DATA_JSON \
  --image_dir $IMAGE_DIR \
  --clip_model_path $CLIP_MODEL_PATH
```
construct.py
```shell
#!/bin/bash
source ~/miniconda3/bin/activate your_env

FOLDER_PATH="./outputs/score"
IMAGE_DIR="./data/images/train2014"
CLIP_ALPHA=0.9
OUTPUT_FILE="./CSR-datasets/your_CSR_dataset.json"

python construct.py \
  --folder_path $FOLDER_PATH \
  --image_dir $IMAGE_DIR \
  --clip_alpha $CLIP_ALPHA \
  --output_file $OUTPUT_FILE
```
