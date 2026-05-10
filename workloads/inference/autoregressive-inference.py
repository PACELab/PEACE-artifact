"""
TODO model - GPT-2, GPT-3, GPT-Neo, GPT-J
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
import signal
#import custom modules
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger

#python autoregressive-inference.py --model_name gpt2-xl --batch_size 1 --output_length 600 --num_requests 10000000 --profile_nstep 2

def inference(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    
    logger = MLLogger(args.log_dir, args, __file__)
    

    logger.log(f"Using transformers library version: {transformers_version}")
    logger.log(f"Using torch version: {torch.__version__}")

    # For gated models (e.g., Gemma, Llama), ensure you are logged in via `huggingface-cli login`
    # or that the HUGGING_FACE_HUB_TOKEN environment variable is set in the execution environment
    # (e.g., passed to the Docker container).
    logger.log("Attempting to load model. For gated models, ensure authentication is set up (e.g., HUGGING_FACE_HUB_TOKEN).")
    #check if HUGGING_FACE_HUB_TOKEN is set
    if "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        logger.log("WARNING: HUGGING_FACE_HUB_TOKEN environment variable is not set. This may cause issues with gated models.")

    #log token
    if os.getenv("HUGGING_FACE_HUB_TOKEN"):
        logger.log(f"HUGGING_FACE_HUB_TOKEN is set: {os.getenv('HUGGING_FACE_HUB_TOKEN')[:10]}... (truncated for security)")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
    except OSError as e:
        logger.log(f"ERROR: Could not load model {args.model_name}. OS Error: {e}")
        logger.log("This might be due to an incorrect model name, network issues, or insufficient permissions (for gated models if token is missing/invalid).")
        exit(1)
    except (KeyError, ValueError) as e:
        logger.log(f"ERROR: Error processing model configuration for {args.model_name}: {e}")
        logger.log(f"This often means the 'transformers' library (version {transformers_version}) is too old or does not support this model type.")
        logger.log("Please try upgrading the 'transformers' library in your environment: pip install --upgrade transformers")
        exit(1)
    except Exception as e:
        logger.log(f"ERROR: An unexpected error occurred while loading model {args.model_name}: {e}")
        exit(1)

    logger.log(f"Successfully loaded model and tokenizer for {args.model_name}")
    logger.log(f"Model parameters: {count_parameters(model)}") # Using your utility

    if "cuda" in str(args.device):
        model.to(args.device)

    # Set the padding token to eos_token if pad_token is not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def countAverageProcessingTime():
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")
        # Write req_processing_time to a csv file
        logger.writecsv(f"req_device{args.device}_batch{args.batch_size}_inference_process_time.csv", [avg_processing_time])

    # Define sigterm handler that executes countAverageProcessingTime()
    def sigterm_handler(signum, frame):
        countAverageProcessingTime()
        logger.log("completed nstep, exiting...")
        exit(0)

    # Handle sigterm signal
    signal.signal(signal.SIGTERM, sigterm_handler)

    # Create a dataset by repeating input_text for num_requests times
    logger.log(f"Creating dataset with {args.num_requests} requests")
    dataset = [args.input_text] * args.num_requests

    model.eval()
    req_processing_time, req_total_time = [], 0
    warmup = 1
    steps = 0
    #total number of tokens generated
    
    # Inference loop over the custom dataset
    for i in range(0, len(dataset), args.batch_size):
        if steps < warmup:
            logger.log(f"warmup steps {steps}")
        if steps == warmup:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
        if steps >= warmup:  
            if args.profile:
                torch.cuda.nvtx.range_push(f"steps{steps}")
                #torch.cuda.nvtx.range_push(f"prefill_and_decode1token_request{steps}")
                

        logger.log(f"Processing data = data {i} to {i + args.batch_size}")
        #if steps >= warmup:  
        #    if args.profile:
        #        torch.cuda.nvtx.range_push(f"moveData{steps}")
                
                

        # Prepare batch of text inputs from the custom dataset
        batch = dataset[i:i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(args.device)

        #if steps >= warmup:
        #    if args.profile:
        #        torch.cuda.nvtx.range_pop()

        start_time = time.time()

        #if steps >= warmup:
        #    if args.profile:
        #        torch.cuda.nvtx.range_push(f"forward{steps}")

        # Token-by-token generation
        generated_tokens = inputs.input_ids
        total_tokens = 0
        for _ in range(args.output_length):
            
            with torch.no_grad():
                outputs = model(generated_tokens)
                logits = outputs.logits
                total_tokens += 1
                

                # Get the last token's logits and predict the next token
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                if steps >= warmup:  
                    if args.profile:
                        if total_tokens == 1:
                            pass
                            #logger.log(f"prefill_and_decode1token_request{steps}poppped")
                            
                            #profile prefill_and_decode1token_request{steps] here
                            #torch.cuda.nvtx.range_pop()
                            #if steps == 10:
                            #   torch.cuda.cudart().cudaProfilerStop()
                # Append the predicted token to the sequence

                if (next_token == tokenizer.eos_token_id).all():
                    break

        if steps >= warmup:
            if args.profile:
                torch.cuda.nvtx.range_pop()

        # Decode the generated text and log predictions
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        logger.log(f"Generated Texts: {generated_texts}")

        req_processing_time.extend([time.time() - start_time] * args.batch_size)
        req_total_time += time.time() - start_time

        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")

        if args.profile_nstep > 0 and i >= args.batch_size * args.profile_nstep:
            if steps >= warmup:
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStop()
            countAverageProcessingTime()
            logger.log(f"completed {i}, exiting...")
            exit(0)

        steps += 1

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    avg_processing_time = np.mean(req_processing_time)
    logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
    logger.log(f"Total processing time: {req_total_time:.4f} seconds")
    # Write req_processing_time to a CSV file
    logger.writecsv(f"req_device{args.device}_batch{args.batch_size}_inference_process_time.csv", req_processing_time)


if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)
    exit(0)
