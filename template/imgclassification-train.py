import sys
import time
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


#import custom functions
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from models.CNN import CNNModel
import torchvision.models as models

def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    print(f"using {args.device}...")
    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        print("[Profile] init phase, starting training...")
        profiler.getMemStats()
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            print(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs

    # Load and preprocess the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model and move it to the GPU if available
    if args.ckpt:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print('check on ckpt model...')
        #TODO actual model should load from pretrain
        
        if args.load_from_ckpt:
            print(f'loading ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt))
        else:
            if "resnet" in args.model_name:
                model = models.resnet50(pretrained=True)
            else:
                model = CNNModel()
    else:
        if "resnet" in args.model_name:
            model = models.resnet50(pretrained=True)
        elif "mobilenet" in args.model_name:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) 
        else:
            model = CNNModel()
    model.to(args.device)
    
    if args.profile:
        print(f"model to cuda...")
        profiler.countModelParameters(model)
        profiler.getModelMemory()
        profiler.getMemStats()
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
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

    #start global time
    start_time = time.time()

    for epoch in range(1, run_epochs+1):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()  # Start time of the current epoch
        print(f"getting batches within epoch{epoch}...")
        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            if args.profile:
                print(f"batch{i}, batch len = {len(inputs)}")
                print(f"send inputs to device, input size={(inputs.nelement()*inputs.element_size() + labels.nelement()*labels.element_size()) / 1024} KB")
                profiler.getMemStats()

            optimizer.zero_grad()
            outputs = model(inputs)
            print("after forward propagation",  profiler.getMemStats()) if args.profile else None
            
            loss = criterion(outputs, labels)
            loss.backward()
            print("after backward pass",  profiler.getMemStats() ) if args.profile else None
            optimizer.step()
            print("after optimizer stepping",  profiler.getMemStats() ) if args.profile else  None

            running_loss += loss.item()
            print("after add running loss",  profiler.getMemStats() ) if args.profile else None
            torch.cuda.reset_peak_memory_stats(device=None) if args.profile else None
            torch.cuda.empty_cache()
        


        epoch_time = time.time() - start_epoch_time
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")
        train_log = {
            "train_loss": running_loss / len(train_dataloader),
        }
        if args.profile:
            print(f"batches done in epoch{epoch}..., reset peak memory")
            profiler.getMemStats()
            torch.cuda.reset_peak_memory_stats(device=None)

        if args.profile:
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

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)
