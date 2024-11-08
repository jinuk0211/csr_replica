
def get_output_texts(
    generation_ids: torch.LongTensor,
    prompt: str,
    generation_tokenizer,
    skip_special_tokens: bool = False,
) -> list[str]:
    generation_texts = generation_tokenizer.batch_decode(
        generation_ids, skip_special_tokens=skip_special_tokens
    )
    output_texts: list[str] = []
    for generation_text in generation_texts:
        generation_text = generation_text.replace(
            "<s> [INST]", "<s>[INST]"
        )  # for llama-2-chat-hf
        split_pieces = generation_text.split(prompt)
        # print(generation_ids)
        # print(generation_tokenizer.decode(generation_ids[0]))
        # print(prompt)
        # print(generation_text)
        # # write to txt:
        # with open('output.txt', 'w') as f:
        #     f.write(generation_text)
        # with open('output2.txt', 'w') as f:
        #     f.write(prompt)
        try:
            assert (
                prompt in generation_text
            ), f"prompt: {prompt} | generation_text: {generation_text}"
            assert (
                len(split_pieces) > 1
            ), f"prompt: {prompt} | generation_text: {generation_text}, {len(split_pieces)}, {split_pieces}"
            output_text = prompt.join(split_pieces[1:])
        except:
            output_text = generation_text[len(prompt) :]
        output_texts.append(output_text)
    return output_texts
def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding

def unpad_output_texts(output_texts: list[str], stop_tokens: list[str]) -> list[str]:
    unpadded_texts: list[str] = []
    for output_text in output_texts:
        for stop_token in stop_tokens:
            output_text = output_text.split(stop_token)[0]
        unpadded_texts.append(output_text)
    return unpadded_texts
  
@torch.inference_mode()
def get_memory_constrained_generation(
    generation_model: transformers.LlamaForCausalLM,
    generation_ids: torch.LongTensor,
    terminators: list[int | None],
    pad_token_id: int | None,
    args,
) -> torch.LongTensor:

    past_key_values = None
    batch_size = generation_ids.shape[0]
    finished_generations = torch.zeros(batch_size).bool().to(generation_model.device)
    while generation_ids.shape[-1] < args.max_tokens:
        try:
            out_dict = generation_model.generate(
                generation_ids,
                pad_token_id=pad_token_id,
                max_new_tokens=1,
                eos_token_id=terminators,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict_in_generate=True,
            )
            if "past_key_values" in out_dict:
                past_key_values = out_dict.past_key_values
            else:
                raise Exception("past_key_values (KV cache) not found in model output")
            generation_ids = out_dict.sequences
        except torch.cuda.OutOfMemoryError:
            break
        just_finished = generation_ids[:, -1] == pad_token_id
        finished_generations = finished_generations | just_finished
        if torch.all(finished_generations):
            break
    return generation_ids



def get_templated_prompt(
    prompt: str,
    llm_name: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> str:
    if "Instruct" in llm_name:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    elif any(s in llm_name for s in ["sft10k", "alpaca-7b", "dpo", "ppo", "human"]):
        templated_prompt = f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    elif "llama-2" in llm_name.lower():
        templated_prompt = f"<s>[INST]\n{prompt} [/INST]"
    else:
        templated_prompt = generation_tokenizer.bos_token + prompt
    return templated_prompt


def write_to_disk(
    all_data: list[dict[str, Any]],
    output_folder: str,
    initial_memory: int,
    pretty_print_output: bool = False,
    record_memory: bool = False,
    force_dump: bool = False,
) -> None:
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    prompt_idx: int = (
        all_data[0]["prompt"]["JSON_idx"]
        if "prompt" in all_data[0]
        and type(all_data[0]["prompt"]) == dict
        and "JSON_idx" in all_data[0]["prompt"]
        else 0
    )
    llm_name: str = all_data[0]["llm_name"]
    reward_model_name: str = all_data[0]["reward_model_name"]
    write_filename = f"{llm_name}_{reward_model_name}_prompt_{prompt_idx:04d}.json"
    write_path = os.path.join(output_folder, write_filename)
    if force_dump or (record_memory and prompt_idx == 0):
        dump_memory_snapshot(write_path, initial_memory)
    if force_dump:
        return
    print_best_trajectory(all_data)
    with open(write_path, "w") as fp:
        if pretty_print_output:
            json.dump(all_data, fp, indent=4)
        else:
            json.dump(all_data, fp)
        print(f"Wrote data to {write_filename}")

def compute_scores(
    question: str,
    output_texts: list[str],
    reward_model_name: str,
    reward_tokenizer,
    reward_model,
) -> list[float]:
    reward_tokens = get_reward_tokens(
        question,
        output_texts,
        reward_model_name,
        reward_tokenizer,
        reward_model.device,
    )
    # print(f"reward_tokens: {reward_tokens}")
    reward_list = get_rewards(reward_model_name, reward_model, reward_tokens)

    if reward_list is None:
        raise Exception("Could not compute scores...")
    return reward_list
def get_memory_constrained_batch_size(length: int, llm_name: str) -> int:
    a, b = get_inverse_function_params(llm_name)
    return int(a / (length + b))


def get_inverse_function_params(llm_name: str) -> tuple[float, float]:
    # NOTE: these parameters are computed by fitting an inverse function to data
    # generated by benchmark_batch_size.py
    if llm_name == "sft10k" or llm_name == "alpaca-7b":
        return (53288.568, 9.164)
    elif llm_name == "Meta-Llama-3-8B":
        return (61626.403, 2.076)
    elif llm_name == "Meta-Llama-3-8B-Instruct" or "Mistral-7B" in llm_name:
        return (61562.069, 2.058)
    else:
        raise Exception("Unknown LLM name")from utils.trajectory import Trajectory

def validate_alpha(alpha: float) -> None:
    if not (0.0 <= alpha < 1.0):
        raise Exception("args.alpha expected to be in [0.0, 1.0)")

from typing import Any


class Trajectory(object):
    def __init__(
        self,
        prompt: str,
        templated_prompt: str,
        padded_output_text: str,
        unpadded_output_text: str,
        score: float,
    ) -> None:
        self.prompt = prompt
        self.templated_prompt = templated_prompt
        self.padded_output_text = padded_output_text
        self.unpadded_output_text = unpadded_output_text
        self.score = score
        self.finished = self.unpadded_output_text != self.padded_output_text

    def get_json_representation(self, sparse: bool = True) -> dict[str, Any]:
        if sparse:
            return {
                "prompt": self.prompt,
                "output": self.unpadded_output_text,
                "score": self.score,
            }
        else:
            return {
                "prompt": self.prompt,
                "templated_prompt": self.templated_prompt,
                "padded_output_text": self.padded_output_text,
                "unpadded_output_text": self.unpadded_output_text,
                "score": self.score,
                "finished": self.finished,
            }

    def get_alpaca_representation(self, generator: str) -> dict[str, str]:
        return {
            "instruction": self.prompt,
            "output": self.unpadded_output_text,
            "generator": generator,
        }