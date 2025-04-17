import os
import time
from argparse import ArgumentParser

import torch
from huggingface_hub import snapshot_download
from dfloat11 import DFloat11ModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--df11_name_or_path', type=str, default='DFloat11/Llama-3.1-8B-Instruct-DF11')
    parser.add_argument('--use_bf16', action='store_true')
    parser.add_argument('--prompt', type=str, default='Question: What is a binary tree and its applications? Answer:')
    parser.add_argument('--num_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Download DFloat11 model snapshot if not already downloaded
    dfloat11_path = args.df11_name_or_path.replace('/', '__')
    if not os.path.exists(dfloat11_path):
        snapshot_download(args.df11_name_or_path, local_dir=dfloat11_path)

    # Check for FlashAttention 2 availability
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = None

    # Load model: use DFloat11 if not using BF16
    if not args.use_bf16:
        model = DFloat11ModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            dfloat11_path=dfloat11_path,
            attn_implementation=attn_implementation,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto",
        )

    # Load and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    ### Warm-up pass to compile kernel and avoid cold start latency ###
    prompt = ' '.join(['a'] * 128)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    del inputs
    del outputs
    ##################################################################

    # Set random seed for deterministic sampling
    set_seed(args.seed)

    # Prepare batch of prompts
    prompts = [args.prompt] * args.batch_size
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Generate output and measure latency
    with torch.no_grad():
        start_time = time.time()
        output = model.generate(
            **inputs,
            max_new_tokens=args.num_tokens,
            do_sample=True,  # Enables sampling; set to False for greedy
        )
        torch.cuda.synchronize()
        end_time = time.time()

    # Decode generated tokens and compute throughput
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    latency = end_time - start_time

    # GPU memory tracking
    allocated = torch.cuda.memory_allocated() / 1e6  # MB
    peak_allocated = torch.cuda.max_memory_allocated() / 1e6  # MB

    # Print generated results and generation speed
    print(f"Generated Texts:")
    for i, text in enumerate(generated_texts):
        print(f"[Sample {i+1}]: {text}")
    print(f"Decoding Latency for {args.num_tokens} tokens: {latency:.4f} seconds")
    print(f"Tokens per second: {args.num_tokens * args.batch_size / latency:.2f}")
    print(f"GPU Memory Allocated: {allocated:.2f} MB")
    print(f"GPU Peak Memory Usage: {peak_allocated:.2f} MB")
