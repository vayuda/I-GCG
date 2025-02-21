# !pip install unsloth vllm
# !pip install --upgrade pillow

"""### Unsloth

Use `PatchFastRL` before all functions to patch GRPO and other RL algorithms!
"""

from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from datasets import load_dataset, Dataset
import re
import json
import os
import torch
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
PatchFastRL("GRPO", FastLanguageModel)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    SYSTEM_PROMPT = """
    You are given a question. You will create as short of a response as possible that will encourage a language model to think less before answering. Do not generate anything else.
    """
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(lambda x: {  # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']},

        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

def create_dataset(create,file_path='cached_dataset.json'):
    if not create:
        return load_dataset_local(file_path)

    dataset = get_gsm8k_questions()
    data_to_save = []
    for item in dataset:
        data_to_save.append({
            'prompt': item['prompt'],
            'answer': item['answer']
        })

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f)
    print(f"Dataset saved to {file_path}")

def load_dataset_local(file_path='cached_dataset.json') -> Dataset:
    """
    Load the dataset from a JSON file
    """
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert the loaded data back to a Dataset object
        dataset = Dataset.from_list(data)
        print(f"Dataset loaded from {file_path}")
        return dataset
    else:
        print("No cached dataset found, creating new dataset...")
        dataset = get_gsm8k_questions()
        save_dataset(dataset, file_path)
        return dataset

def deepseek_token_reward_func(r1_tokenizer, r1_model):
    def f(prompts, completions, **kwargs) -> list[float]:
        """
        For each prompt, this function:
        1. Extracts the original GSM8k question from the last message.
        2. Extracts the adversarial suffix from the corresponding completion (assumed to be in completion[0]['content']).
        3. Appends the suffix to the question and feeds the combined prompt into the DeepSeek r1 model.
        4. Extracts the <think> ... </think> section from the generated output.
        5. Computes the reward as the number of tokens in the extracted thinking portion.

        Args:
            prompts (list): A list of prompts, where each prompt is a list of message dictionaries.
            completions (list): A list of completions corresponding to the adversarial suffix.
            **kwargs: Additional keyword arguments if needed.

        Returns:
            list[float]: A list of rewards (token counts of the thinking portion) for each prompt.
        """
        rewards = []
        # Get the device of the r1 model
        device = next(r1_model.parameters()).device

        for prompt_messages, completion in zip(prompts, completions):
            # Extract the original question (assumed to be the last message in the prompt)
            question = prompt_messages[-1]['content']
            # Extract adversarial suffix from the completion (assumed to be the first message's content)
            adversarial_suffix = completion[0]['content']
            print(completion[0])
            print(adversarial_suffix)
            combined_prompt = f"{question} {adversarial_suffix}"

            # Tokenize the combined prompt and move tensors to the same device as r1_model.
            inputs = r1_tokenizer(combined_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate output from the r1 model.
            output_ids = r1_model.generate(
                **inputs,
                max_new_tokens=1024,  # Adjust as needed.
                do_sample=True,
                temperature=0.8,
            )
            generated_text = r1_tokenizer.decode(
                output_ids[0], skip_special_tokens=True)

            # Extract the thinking portion from the output using regex.
            # NOTE: We now use a capturing group to capture text between <think> and </think>
            think_pattern = r"<think>(.*?)</think>"
            match = re.search(think_pattern, generated_text, re.DOTALL)
            if match:
                thinking_text = match.group(1).strip()
            else:
                thinking_text = ""

            # Count tokens in the thinking text.
            token_count = len(r1_tokenizer.tokenize(thinking_text))
            # You can choose to reward higher token count (or penalize with a negative sign) as needed.
            reward = -token_count
            rewards.append(reward)

        return rewards

    return f
def main():
    dataset = create_dataset(False)
    if dataset is None:
        raise ValueError("Failed to create the dataset.")
    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower
    training_args = GRPOConfig(
        use_vllm=False,  # use vLLM for fast inference!
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=8,  # Decrease if out of memory
        max_prompt_length=256,
        max_completion_length=200,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
    )

    # adversarieal model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,  # Reduce if out of memory
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    r1_model, r1_tokenizer = FastLanguageModel.from_pretrained(
        # Replace with your DeepSeek r1 model identifier
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.5,
    )
    r1_model = FastLanguageModel.for_inference(r1_model)
    reward_func = deepseek_token_reward_func(r1_tokenizer, r1_model)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_lora("grpo_saved_lora")


if __name__ == "__main__":
    main()

# Merge to 16bit
# if False:
#     model.save_pretrained_merged(
#         "model", tokenizer, save_method="merged_16bit",)
# if False:
#     model.push_to_hub_merged("hf/model", tokenizer,
#                              save_method="merged_16bit", token="")

# # Merge to 4bit
# if False:
#     model.save_pretrained_merged(
#         "model", tokenizer, save_method="merged_4bit",)
# if False:
#     model.push_to_hub_merged("hf/model", tokenizer,
#                              save_method="merged_4bit", token="")

# # Just LoRA adapters
# if False:
#     model.save_pretrained_merged("model", tokenizer, save_method="lora",)
# if False:
#     model.push_to_hub_merged("hf/model", tokenizer,
#                              save_method="lora", token="")
# # Save to 8bit Q8_0
# if False:
#     model.save_pretrained_gguf("model", tokenizer,)
# # Remember to go to https://huggingface.co/settings/tokens for a token!
# # And change hf to your username!
# if False:
#     model.push_to_hub_gguf("hf/model", tokenizer, token="")

# # Save to 16bit GGUF
# if False:
#     model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
# if False:
#     model.push_to_hub_gguf("hf/model", tokenizer,
#                            quantization_method="f16", token="")

# # Save to q4_k_m GGUF
# if False:
#     model.save_pretrained_gguf(
#         "model", tokenizer, quantization_method="q4_k_m")
# if False:
#     model.push_to_hub_gguf("hf/model", tokenizer,
#                            quantization_method="q4_k_m", token="")

# # Save to multiple GGUF options - much faster if you want multiple!
# if False:
#     model.push_to_hub_gguf(
#         "hf/model",  # Change hf to your username!
#         tokenizer,
#         quantization_method=["q4_k_m", "q8_0", "q5_k_m",],
#         token="",
#     )
