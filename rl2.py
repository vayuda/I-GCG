import re
import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

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
    You are given a question. You will create as short of a response as possible that will encourage a language model to think less before answering. Do not generate anything else. Do not answer the question.
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

def create_dataset(create, file_path='cached_dataset.json'):
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
    return dataset

def load_dataset_local(file_path='cached_dataset.json') -> Dataset:
    """
    Load the dataset from a JSON file.
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
        # Save the dataset for future use.
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f)
        return dataset

def deepseek_token_reward_func(r1_tokenizer, r1_model):
    def f(prompts, completions, **kwargs) -> list[float]:
        """
        For each prompt, this function:
        1. Extracts the original GSM8k question from the last message.
        2. Extracts the adversarial suffix from the corresponding completion
           (assumed to be in completion[0]['content']).
        3. Appends the suffix to the question and feeds the combined prompt
           into the DeepSeek r1 model.
        4. Extracts the <think> ... </think> section from the generated output.
        5. Computes the reward as the negative number of tokens in the thinking portion.

        Args:
            prompts (list): A list of prompts, where each prompt is a list of message dictionaries.
            completions (list): A list of completions corresponding to the adversarial suffix.
            **kwargs: Additional keyword arguments if needed.

        Returns:
            list[float]: A list of rewards for each prompt.
        """
        rewards = []
        # Get the device of the r1 model
        device = next(r1_model.parameters()).device

        for prompt_messages, completion in zip(prompts, completions):
            # Extract the original question (assumed to be the last message in the prompt)
            question = prompt_messages[-1]['content']
            # Extract adversarial suffix from the completion (assumed to be the first message's content)
            adversarial_suffix = completion[0]['content']
            combined_prompt = f"{question} {adversarial_suffix}"
            # print('\n\n\n ======= VERIFY ======', adversarial_suffix, '======= VERIFY ======\n\n\n')

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
            generated_text = r1_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract the thinking portion from the output using regex.
            think_pattern = r"<think>(.*?)</think>"
            match = re.search(think_pattern, generated_text, re.DOTALL)
            if match:
                thinking_text = match.group(1).strip()
            else:
                thinking_text = ""

            # Count tokens in the thinking text.
            token_count = len(r1_tokenizer.tokenize(thinking_text))
            reward = -token_count
            rewards.append(reward)

        return rewards

    return f

def main():

    # Create or load the dataset.
    dataset = create_dataset(True)
    if dataset is None:
        raise ValueError("Failed to create or load the dataset.")

    max_seq_length = 1024  # Can increase for longer reasoning traces
    lora_rank = 64  # Larger rank = smarter, but slower

    # Configure training arguments.
    training_args = GRPOConfig(
        use_vllm=False,  # vLLM is not used here.
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16= not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # Increase for smoother training if needed.
        num_generations=8,  # Decrease if out of memory.
        max_prompt_length=256,
        max_completion_length=200,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",  # Set to "wandb" if using Weights & Biases.
        output_dir="outputs",
    )

    # Initialize the Accelerator.
    accelerator = Accelerator()

    # ===== Load adversarial model in 16-bit precision =====
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    # Load the model using 16-bit precision instead of 4-bit quantization.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=max_seq_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Apply LoRA adapters using peft.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    # Enable gradient checkpointing if desired (native HF method)
    model.gradient_checkpointing_enable()

    # ===== Load DeepSeek r1 model =====
    r1_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    r1_model = AutoModelForCausalLM.from_pretrained(
        r1_model_name,
        device_map="auto",
        max_length=max_seq_length,
    )
    r1_tokenizer = AutoTokenizer.from_pretrained(r1_model_name, use_fast=True)
    r1_model.eval()

    # Prepare models with Accelerator.
    model = accelerator.prepare(model)
    r1_model = accelerator.prepare(r1_model)

    # Create the reward function using the r1 model/tokenizer.
    reward_func = deepseek_token_reward_func(r1_tokenizer, r1_model)

    # Create and run the GRPO trainer.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Save the resulting LoRA adapters.
    model.save_pretrained("grpo_saved_lora")
    print("LoRA adapters saved to 'grpo_saved_lora'.")


if __name__ == "__main__":
    main()
