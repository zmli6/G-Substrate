import argparse
import json
import os
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--raw_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--template", type=str, default="qwen3_vl_nothink")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    print(f"[INFO] Loading model from {args.model_path}...")
    # Initialize vLLM
    # vLLM automatically enables PagedAttention and KV Cache management.
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=0.95,
        max_model_len=16384, # Adjust based on your needs
        limit_mm_per_prompt={"image": 1}, # Optimization for multi-modal models
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    print(f"[INFO] Loading data from {args.raw_file}...")
    raw_data = load_json(args.raw_file)
    
    prompts = []
    inputs_data = []

    # Prepare prompts
    for item in raw_data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        images = item.get('images', [])
        
        # Construct content
        content = instruction
        if input_text:
            content += "\n" + input_text

        messages = []
        
        # Handle Images & Template Logic
        # Note: This is a simplified logic. LLaMA-Factory has more complex handling.
        if images and len(images) > 0:
            image_path = images[0]
            # Ensure absolute path if needed, or assume paths in json are correct
            
            if "qwen" in args.template.lower():
                # Qwen format
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": content}
                    ]
                })
            elif "intern" in args.template.lower():
                # InternVL format
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": content}
                    ]
                })
            else:
                # Fallback
                messages.append({"role": "user", "content": content})
        else:
            # Text only
            messages.append({"role": "user", "content": content})

        # Apply chat template
        try:
            # We use the tokenizer's chat template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"[WARN] Template application failed: {e}. Using raw content.")
            prompt = content

        prompts.append(prompt)
        # For vLLM with images, we might need to pass multi_modal_data if not embedded in prompt text
        # But recent vLLM / HF tokenizers handle <image> tags in text.
        # If vLLM fails to load images from prompt text, we might need explicit multi_modal_data.
        # For now, assuming tokenizer handles the <image> tag insertion.
        
        inputs_data.append(item)

    print(f"[INFO] Generating for {len(prompts)} samples...")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    # vLLM handles batching and KV caching internally here
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        result_item = inputs_data[i].copy()
        result_item['predict'] = generated_text
        results.append(result_item)

    print(f"[INFO] Saving results to {args.output_file}...")
    save_jsonl(results, args.output_file)
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
