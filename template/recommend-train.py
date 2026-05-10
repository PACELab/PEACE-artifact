import os
import sys
import time
import argparse
import evaluate
import numpy as np

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm

#import custom functions
sys.path.append('../')
from utils.util import count_parameters, draw_mem_summary
from utils.parser import MLParser
from utils.profiler import MLProfiler


def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    print(f"using {args.device}...")
    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            print(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs
        
    """
    TODO Count data download & process time
    """
    dataset = load_dataset(args.data)
    dataset["train"][100]

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)

    """
    end TODO of Counting data download & process time
    """
    if args.ckpt:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print('check on ckpt model...')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
        if args.load_from_ckpt:
            print(f'loading ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    model.to(args.device)

    if args.profile:
        print(f"model to cuda...")
        profiler.countModelParameters(model)
        profiler.getModelMemory()
        global_start_time = time.time()
        profiler.getMemStats(ts=time.time()-global_start_time, state='load_model')
        

    optimizer = AdamW(model.parameters(), lr=args.lr)


    num_training_steps = args.n_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    #import wandb
    if args.wandb_logging:
        import wandb

        #wandb.init(project=f"{os.path.basename(__file__)[:-3]}_epochs_{args.n_epoch}_batch_size_{args.batch_size}_profile_earlystop_{args.profile}_profile_epochs_{args.profile_run}", entity="nba556677go", name=args.exp_name, config=args)
        wandb.init(
        # set the wandb project where this run will be logged
        entity="nba556677go",
        project="k8s-scheduling",
        name=args.exp_name,
        # track hyperparameters and run metadata
        config=args  
        )

        wandb.watch(model)

    progress_bar = tqdm(range(num_training_steps))


    start_time = time.time()

    model.train()
    train_start_time = time.time()
    for epoch in range(1, run_epochs+1):

        start_epoch_time = time.time()
        running_loss = 0.0
        n_batch = 0
        for batch in train_dataloader:
            n_batch += 1
            
                
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if args.profile:
                print(f"batch{n_batch}, batch_len = {len(batch)}")
                input_size = 0
                for _, v in batch.items():
                    #print(v)
                    input_size += v.nelement()*v.element_size()

                print(f"send inputs to device, input size={input_size/1024**2} MB")
                del input_size
                profiler.getMemStats(ts=time.time()-global_start_time,state='input_data')
            
            outputs = model(**batch)
            print("after forward propagation")
            profiler.getMemStats(ts=time.time()-global_start_time,state='fwd') if args.profile else None
            
            loss = outputs.loss
            loss.backward()
            print("after backward pass")
            profiler.getMemStats(ts=time.time()-global_start_time,state='bwd') if args.profile else None
            optimizer.step()
            print("after optimizer stepping")
            profiler.getMemStats(ts=time.time()-global_start_time,state='optimizer_step') if args.profile else None
            lr_scheduler.step()
            #print("after lr_scheduler stepping",  profiler.getMemStats() ) if args.profile else  None
            optimizer.zero_grad()
            progress_bar.update(1)

            running_loss += loss.item()
            torch.cuda.reset_peak_memory_stats(device=None) if args.profile else None
            del loss
            del outputs
            if n_batch == 10: 
                break
            

        
        epoch_time = time.time() - start_epoch_time
        train_log = {
            "train_loss": running_loss / len(train_dataloader),
        }
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")
            
        if args.profile:
            print(f"batches done in epoch{epoch}..., reset peak memory")
            #profiler.getMemStats(ts=time.time()-global_start_time,state='epoch_done') if args.profile else None
            torch.cuda.reset_peak_memory_stats(device=None)
            GPUinfo = profiler.getSMIinfobyTask(sys.argv)
            print(f"[Profile]GPU INFO - {GPUinfo}")
            gpuInfoDict = list(GPUinfo.values())[0][0]
            profile_log = {
                "GPUMem (MB)" : int(gpuInfoDict["gpu_mem"][:-3]),
                "CPUpercentage" : float(gpuInfoDict["cpu"]),
                "RAMpercentage" : float(gpuInfoDict["mem"]),
                "Time elapsed": epoch_time,

            }
            profiler.addEpochTime(epoch_time)
            if args.wandb_logging:
                wandb.log({**profile_log, "epoch": epoch})

        if args.wandb_logging:
            wandb.log({**train_log, "epoch": epoch})


            
        

    total_training_time = time.time() - start_time
    m, s = divmod(total_training_time, 60)
    h, m = divmod(m, 60)
    print(f"Total Training Time: {h:f}h {m:02f}m {s:02f}s")
    training_time_per_epoch = total_training_time / run_epochs
    m, s = divmod(training_time_per_epoch , 60)
    h, m = divmod(m, 60)
    print(f"Average Time for each Epoch: {h:f}h {m:02f}m {s:02f}s")

    if args.wandb_logging:
        wandb.log({"Total Training Time": total_training_time,
                    "Average Time for each Epoch": training_time_per_epoch,
                    })
    if args.profile and not args.profile_all_estimation:
        profiler.drawMemSummary(output=f'{args.model_name}_steps_{n_batch}_batchSize_{args.batch_size}_memgraph.jpg')
        profiler.saveEarlyStop(os.path.basename(__file__)[:-3], GPUinfo, training_time_per_epoch)
        print(f"Earlystop at epoch{run_epochs}")

    """
    TODO sample for saving check point model - for future preemption purpose 
    """
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt_dir / f"model_{args.device}_{args.exp_name}.pt")
        print(f"{'':30s}*** Best model saved ***")


    ''' 
    Profiling task after training
    '''
    if args.profile and args.profile_all_estimation:
        result_csv = profiler.saveAllEpochEstimatedTime(total_training_time, training_time_per_epoch)
        cwd = os.getcwd()
        profiler.draw_summary(f"{result_csv}")
        print(f'Current working directory is {cwd}')
        print("Profile finished!")
            
    print("Training finished!")


    metric = evaluate.load("accuracy")
    model.eval()


    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)
    exit(0)