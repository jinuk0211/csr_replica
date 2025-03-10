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
```python
from utils import *
from accelerate.utils import gather_object
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer
import json
from accelerate import Accelerator
from PIL import Image
import os
import argparse


def eval_model(args):
    accelerator = Accelerator()
    model_path = args.model_path
    base_hf_model_path = args.base_hf_model_path
    mapping_path = args.weight_mapping_path
    output_dir = args.output_dir

    # Load Model
    processor = AutoProcessor.from_pretrained(base_hf_model_path)
    if args.is_hf:
        model, model_tokenizer, base_tokenizer = load_hf_llava_model(model_path)
    else:
        model, model_tokenizer, base_tokenizer = load_llava_model(model_path, base_hf_model_path, mapping_path)
    model.to(accelerator.device)
#------------------------------------------
def load_hf_llava_model(model_path):
    # Weights are loaded directly with hf-llava version
    model = LlavaForConditionalGeneration.from_pretrained(model_path, device_map='cpu', torch_dtype=torch.float16)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side='left') 
    return model, model_tokenizer, base_tokenizer
def load_llava_model(model_path, base_hf_model_path, mapping_path):
    # Weights should be specially loaded with other llava versions
    model = LlavaForConditionalGeneration.from_pretrained(base_hf_model_path, device_map='cpu', torch_dtype=torch.float16)
    model_tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_tokenizer = AutoTokenizer.from_pretrained(base_hf_model_path, use_fast=False, padding_side='left')
    state_dicts = load_and_merge_models(model_path)
    with open(mapping_path, 'r', encoding='utf-8') as f1:
        mapping_keys = json.load(f1)

    modified_weights = {}
    for old_key, value in state_dicts.items():
        new_key = mapping_keys.get(old_key, old_key)
        modified_weights[new_key] = value
    modified_weights['language_model.model.embed_tokens.weight'] = model.state_dict()['language_model.model.embed_tokens.weight']
    modified_weights['language_model.lm_head.weight'] = model.state_dict()['language_model.lm_head.weight']
    model.load_state_dict(modified_weights, strict=True)
    return model, model_tokenizer, base_tokenizer
#-------------------------------

    # Load Dataset
    with open(args.dataset_path, 'r', encoding='utf8') as fp:
        my_dataset = json.load(fp)
    llava_loader = get_llava_dataloader(my_dataset, 1)
    llava_loader, processor = accelerator.prepare(llava_loader, processor)

    with torch.no_grad():
        for data in llava_loader:
            input_questions = data['input']
            input_questions = [q.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "") for q in input_questions]
            image_paths = data['image']
            qid = data['question_ids']
            images = []

            for image_path in image_paths:
                images.append(Image.open(os.path.join(args.images_dir, 'COCO_train2014_' + image_path)))

            prompts = get_prompts(input_questions)
            sentence_end_id = int(args.period_id)
            max_length = int(args.max_length)
            token_level_beams = int(args.num_token_beams)
            max_new_tokens = int(args.max_new_tokens)
            diversity_penalty = float(args.diversity_penalty)
            num_beams = int(args.num_beams)
            num_beam_group = int(args.num_beam_group)

            # Batched inference is not supported yet
            result = gather_object(sentence_level_beam_search_tree(
                qid[0],model,accelerator,processor,base_tokenizer,
                model_tokenizer,prompts[0],images[0],sentence_end_id,
                max_length,max_new_tokens,num_beams,num_beam_group,
                token_level_beams,diversity_penalty))

            if accelerator.is_main_process:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for obj in result:
                    save_path = os.path.join(output_dir, str(obj['id']) + '.pkl')
                    save_object(obj, save_path)

            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()

def sentence_level_beam_search_tree(qid, model, accelerator, processor, tokenizer, after_tokenizer, initial_text, images, sentence_end_id, max_length, max_new_tokens, num_beams, num_beam_group, token_level_beams, diversity_penalty):
    root = Node(initial_text, 0, 0) #self, text, score, depth, parent=None, is_final=False
    active_nodes = [root]

    with torch.no_grad():
        while active_nodes:
            new_nodes = []

            for node in active_nodes:
                inputs = processor(text=node.text, images=images, return_tensors="pt").to(model.device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        num_beams=token_level_beams,
                        eos_token_id=sentence_end_id,
                        num_beam_groups=num_beam_group,
                        diversity_penalty=diversity_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        num_return_sequences=token_level_beams,
                        max_new_tokens=max_new_tokens,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1]:]
                gen_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                for j, (text, score) in enumerate(zip(gen_texts, outputs.sequences_scores)):
                    new_score = node.score + score.item()
                    is_final = (tokenizer.eos_token_id in gen_sequences[j].tolist()) or (after_tokenizer.eos_token_id in gen_sequences[j].tolist() or len(tokenizer.decode(outputs.sequences[j])) >= max_length)
                    new_node = Node(text, new_score, node.depth + 1, node, is_final)
                    node.add_child(new_node)

                    if not is_final:
                        new_nodes.append(new_node)

            new_nodes.sort(key=lambda x: x.score, reverse=True)
            active_nodes = new_nodes[:int(num_beams/2)-1] + new_nodes[-int(num_beams/2):] if len(new_nodes) >= num_beams else new_nodes

            if not active_nodes:
                break

    return [{'id': qid, 'tree': root}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to your model")
    parser.add_argument("--base_hf_model_path", type=str, default='llava-hf/llava-1.5-7b-hf', help="Path to huggingface base model")
    parser.add_argument("--is_hf", type=int, default=1, help="If it's a hf model")
    parser.add_argument("--dataset_path", type=str, default='./data/CSR-Prompt-Dataset-12k.json', help="Path to the prompt dataset")
    parser.add_argument("--images_dir", type=str, default="./data/images/train2014", help="Directory to images")
    parser.add_argument("--output_dir", type=str, default="./outputs/sample", help="Path to step1's result")
    parser.add_argument("--diversity_penalty", type=float, default=3.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_beam_group", type=int, default=5)
    parser.add_argument("--num_token_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--period_id", type=int, default=29889)
    parser.add_argument("--weight_mapping_path", type=str, default='./model_mapping/key_mapping_hf_7b.json', help="To load non-hf model specially")
    args = parser.parse_args()

    eval_model(args)
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
```python
import os
import json
from transformers import CLIPModel, AutoProcessor
import torch
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import Node, Rank_Node, extract_new_text, load_and_store_pkl_files, clean_tree, save_pickle


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn(batch):
    return batch


def get_clip_score(new_text, image, model, processor):
    if not new_text:
        return None
    inputs = processor(text=[new_text], images=image, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    clip_score = logits_per_image.cpu().detach().numpy()[0][0]
    return clip_score


def dfs_score(node, model, processor, parent=None, image=None):
    if image is None:
        raise ValueError("Image must be provided")

    new_text = extract_new_text(node.text, parent.text if parent else None)
    clip_score = get_clip_score(new_text, image, model, processor) if parent else None

    rank_node = Rank_Node(
        text=node.text,score=node.score,
        depth=node.depth,parent=parent,
        is_final=node.is_final,clip_score=clip_score
    )

    if parent:
        parent.add_child(rank_node)

    for child in node.children:
        child_len = len(extract_new_text(child.text, node.text))
        if child_len >= 4:
            dfs_score(child, model, processor, rank_node, image)

    return rank_node


def get_result(qid, tree, clip_model, clip_processor, image):
    new_tree = dfs_score(tree, clip_model, clip_processor, None, image=image)
    new_tree.calculate_ranks()
    return [{'qid': qid, 'tree': new_tree}]


def eval_model(args):
    folder_path = args.folder_path
    pkl_data_list = load_and_store_pkl_files(folder_path)
    output_dir = args.output_dir

    with open(args.data_json, 'r') as file:
        data = json.load(file)

    id_image_map = {item['id']: item['image'] for item in data}

    list_dataset = ListDataset(pkl_data_list)
    dataloader = DataLoader(list_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    image_dir = args.image_dir
    clip_model = CLIPModel.from_pretrained(args.clip_model_path)
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)
    accelerator = Accelerator()
    clip_model, clip_processor, dataloader = accelerator.prepare(clip_model, clip_processor, dataloader)

    for tree_dict in dataloader:
        tree_dict = tree_dict[0]
        qid = tree_dict['id']
        tree = clean_tree(tree_dict['tree'])
        img_path = id_image_map[qid]
        image = Image.open(os.path.join(image_dir, 'COCO_train2014_' + img_path))
        with torch.no_grad():
            result = gather_object(get_result(qid, tree, clip_model, clip_processor, image))

        if accelerator.is_main_process:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for obj in result:
                save_path = os.path.join(output_dir, str(obj['qid']) + '.pkl')
                save_pickle(obj, save_path)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Directory to the step1's .pkl results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save step2's .pkl results")
    parser.add_argument("--data_json", type=str, required=True, help="Path to the JSON data file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--clip_model_path", type=str, default='openai/clip-vit-large-patch14-336', help="Path to the CLIP model")
    args = parser.parse_args()

    eval_model(args)
```
construct.py
```python
import os
import re
import json
import argparse
from utils import load_pickles, Rank_Node


def dfs(node, path=[], cumulative_score=0, clip_alpha=0.8):
    if node.rank is not None and node.clip_rank is not None:
        cumulative_score += (1-clip_alpha)*node.rank + clip_alpha * node.clip_rank
    current_path = path + [(node.text, cumulative_score)]
    if node.is_final:
        return [(current_path, cumulative_score)]
    paths_scores = []

    for child in node.children:
        paths_scores.extend(dfs(child, current_path, cumulative_score, clip_alpha))
    return paths_scores


def process_data(args):
    folder_path = args.folder_path
    image_dir = args.image_dir
    clip_alpha = args.clip_alpha
    output_file = args.output_file
    tree_list = load_pickles(folder_path)
    data_list = []
    data_list_with_score = []

    for tree_dict in tree_list:
        this_id_dict = {}
        qid = tree_dict['qid']
        tree = tree_dict['tree']

        tree.calculate_ranks()

        img_path = str(qid)+'.jpg'
        results = dfs(tree, clip_alpha=clip_alpha)
        sorted_results = sorted(results, key=lambda x: x[1] / len(x[0]))
        chosen_process = sorted_results[0][0]
        rejected_process = sorted_results[-1][0]

        the_input = chosen_process[0][0].strip()
        pattern = r"USER:\s*<image>\s*"
        replacement = "USER: <image>"

        chosen = re.sub(pattern, replacement, chosen_process[-1][0])
        rejected = re.sub(pattern, replacement, rejected_process[-1][0])
        chosen = chosen[len(the_input):].strip()
        rejected = rejected[len(the_input):].strip()

        chosen_conv = [{'from': 'human', 'value': the_input}, {'from': 'gpt', 'value': chosen}]
        rejected_conv = [{'from': 'human', 'value': the_input}, {'from': 'gpt', 'value': rejected}]

        this_id_dict['id'] = qid
        this_id_dict['image'] = os.path.join(image_dir, 'COCO_train2014_' + img_path)
        this_id_dict['conversations'] = chosen_conv
        this_id_dict['rejected_conversations'] = rejected_conv

        data_list.append(this_id_dict)
        data_list_with_score.append((this_id_dict, chosen_process[-1][1] - rejected_process[-1][1]))

    with open(output_file, mode='w') as json_file:
        json.dump(data_list, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Directory to save step2's .pkl results")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--clip_alpha", type=float, default=0.9, help="Alpha value for CLIP")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSR JSON dataset")
    args = parser.parse_args()

    process_data(args)
```
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
