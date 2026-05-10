import argparse
import time
from utils.predictor import Predictor
from utils.trainer import Trainer
from utils.loaddata import getstage1trainData, train_test_split_with_counts, getcloud_stage1trainData, get_multiinstance_stage1trainData, Dataloader
from pathlib import Path
from datetime import datetime
import logging

_script_start_time = time.time()

def predict(args, testingfile, model, excluded_cols):
    
    #stage 1
    predictor = Predictor(args = args, 
                        modelpath=model,
                        stage1_data = testingfile,
                        stage2_data="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/stage2/baseline_steps_stage2.csv",
                        actual_share_throughput_data=[args.sharedThroughputData],
                        excluded_cols = excluded_cols)
    
    test_dataloader = Dataloader(args, 
                                    baselineData="./tests/mps/analysis/1222_baseline_metrics.csv",
                                    kernelData="",
                                    shareThroughputData="",
                                    sharePowerData="",
                                    shareDurationData="",
                                    shareEnergyData="")
    
    if args.predAllacc:
        assert(args.train_postprocess_file)
        t_data_start = time.time()
        X_test, y_test , test_workloads, columns_excluded, (test_mean, test_std) =  test_dataloader.aggregate_and_split_train_test(data=args.test_file,
                                                train_postprocess_data=args.train_postprocess_file,
                                                test_workload="",
                                                mode="test",
                                                target=args.label_policy,
                                                correlation=args.correlation,
                                                n_combination=args.n_combination)
        t_data_end = time.time()
        data_process_time = t_data_end - t_data_start

        t_pred_start = time.time()
        best_pair = predictor.predictAllWorkloads(modelpath = model, HPworkload="", 
                                        X_test=X_test, y_test=y_test, 
                                        target=args.label_policy, 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
        t_pred_end = time.time()
        predict_time = t_pred_end - t_pred_start

        end_to_end_time = t_pred_end - _script_start_time

        timing_file = f"{args.output_dir}/pred_timing_{args.label_policy}.txt"
        with open(timing_file, "w") as f:
            f.write(f"data_process_time_sec: {data_process_time:.4f}\n")
            f.write(f"predict_time_sec: {predict_time:.4f}\n")
            f.write(f"end_to_end_time_sec: {end_to_end_time:.4f}\n")
        print(f"Timing saved to {timing_file}")

        exit(0)
    if args.end_to_end: 
        
        predictor.predictE2E( test_file = args.test_file, 
                        policy=args.label_policy, throughput_model=args.throughput_model,
                         power_model=args.power_model, randomseed=30)
        exit(0)
        
    if args.pred_multiinstance:
        best_pair = predictor.predictAll_multiinstance_Workloads(HPworkload=args.HPworkload, 
                                        target="sum_throughput", 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
        exit(0)
    #stage1
    best_pair = predictor.predictPair(HPworkload=args.HPworkload, 
                                        target="sum_throughput", 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
    
    print(best_pair)
    
        
    
    #stage2
    #best_thread = predictor.getBestThread(policy="max-min", pair=best_pair)
    

    #plot results
    strbest_pair = "-".join(best_pair)
    predictor.plot_results(outname=f"{args.output_dir}/HP-{args.HPworkload}_MPS{args.targetMPS}.png", 
                           pair=best_pair)


def train(args):
    print("start training...")
    
    
    trainer = Trainer(args)
    excluded_cols = trainer.processData(data=args.train_file, target="threadclass")
    #selected_feats=["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio"]

    
    modelfile = trainer.train(modeltype=args.modeltype)
    print(f"model saved to {modelfile}")
    #save modelpath to 
    return modelfile, excluded_cols

def getstage1Data(args):
    if args.modeltype == "hotcloud":
        """
        datafile = getcloud_stage1trainData(baselineData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_baseline_labels.csv",
                                            shareThroughputData=args.sharedThroughputData,
                                            kernelData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_kernel_labels.csv",
                                            outname="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_labels.csv",
                                            targetMPS=args.targetMPS,
                                            selected_feats=["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"],
                                            n_combination=args.n_combination)
        """
        datafile = get_multiinstance_stage1trainData(baselineData="/home/cc/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_baseline_labels.csv", 
                       shareThroughputData=args.sharedThroughputData, 
                       kernelData="/home/cc/mlProfiler/tests/mps/multiinstance/hotcloud_0912_gptxl_kernel_labels_comb2_batches2-8-16.csv",
                       outname=f"./tests/mps/multiinstance/dataset/hotcloud/0912hotcloud_gpt2xl_batchedthroughput_total_labels_comb{args.n_combination}_batch2-8", 
                       targetMPS=args.targetMPS,
                       n_combination=args.n_combination,
                       hotcloud=True)
    
    else:
        datafile = get_multiinstance_stage1trainData(baselineData="", 
                       shareThroughputData=args.sharedThroughputData, 
                       kernelData="./tests/mps/multiinstance/0206_baseline_labels_comb2_batches2.csv",
                       outname=f"./tests/mps/multiinstance/dataset/{args.modeltype}/1222_batchedthroughput_total_labels_comb{args.n_combination}", 
                       targetMPS=args.targetMPS,
                       n_combination=args.n_combination)
    
    
    return datafile

def getstage2Data(args):
    dataloader = Dataloader(args,
                            baselineData=str(args.baseline_file),
                            #baselineData="./tests/mps/analysis/0206_baseline_metrics.csv",
                            #freq 1530
                            kernelData="./tests/mps/multiinstance/02062025_freq1530_baseline_labels_comb2_batches2_cuda_samples_0515.csv",
                            ########################################################
                            #[SOCC'26 Rebuttal ONLY - triton kernel data]
                            #tritonkernel still use original 1530 metrics. not triton baseline metrics.[NOT GOOD]
                            #kernelData="./tests/mps/multiinstance/02062025_freq1530_baseline_labels_comb2_batches2_triton_baseline_NOTRITON_0515.csv",
                            #[REBUTTAL USED]freq1530 triton kernel with triton baseline metrics 
                            #kernelData="./tests/mps/multiinstance/04132026_triton_DL_baseline_labels_comb2_batches2.csv",
                            ########################################################
                            #freq 900
                            #kernelData="./tests/mps/multiinstance/05022025_freq900_DL_baseline_labels_comb2_batches2.csv",
                            #kernelData="./tests/mps/multiinstance/05022025_freq900_mergecudaDL_removebothcudasamples_baseline_labels_comb2_batches2.csv",
                            #freq 300
                            #kernelData="./tests/mps/multiinstance/05052025_freq300_mergecudaDL_baseline_labels_comb2_batches2_removebothcudasamples.csv",
                            #kernelData="./tests/mps/multiinstance/05052025_freq300_DL_baseline_labels_comb2_batches2.csv",
                            #comb3
                            #freq 1530 
                            #kernelData="./tests/mps/multiinstance/09132025_freq1530_mergecudaDL_baseline_labels_comb3_batches2_removebothcudasamples.csv",
                            #freq 900
                            #kernelData="./tests/mps/multiinstance/09132025_freq900_mergecudaDL_baseline_labels_comb3_batches2_removebothcudasamples.csv",
                            #freq 300
                            #kernelData="./tests/mps/multiinstance/09132025_freq300_mergecudaDL_baseline_labels_comb3_batches2_removebothcudasamples.csv",
                            shareThroughputData=args.sharedThroughputData,
                            sharePowerData=args.sharedPowerData,
                            shareDurationData=args.sharedDurationData,
                            shareEnergyData=args.sharedEnergyData,
                            )
    datafile = dataloader.getstage2trainData(
                                outname=f"{args.output_dir}/{args.output_prefix}_throughput_total_labels_comb{args.n_combination}", 
                            )
    


if __name__ == "__main__":
    #stage1Data = getstage1Data()
    #argparse
    #add argparse



    # Get today's date in the MMDD format
    today_date = datetime.now().strftime("%m%d")
    parser = argparse.ArgumentParser()

    #common
    parser.add_argument("--output_dir","-o", type=Path, 
                        help="output directory of training model and predictions") 
    #outfile prefix, default is today's date

    parser.add_argument("--output_prefix",
                        default=f"{today_date}", 
                        type=str, help="output file prefix")
    #for predictions
    parser.add_argument("--train_postprocess_file", "-tp", type=Path)
    parser.add_argument("--model", "-m",type=Path)
    parser.add_argument("--HPworkload", "-w", type=str)
    parser.add_argument("--targetMPS", "-t", type=int)
    parser.add_argument("--end_to_end", "-e2e", action="store_true")
    #add throughput_model
    parser.add_argument("--throughput_model", "-tm", type=Path)
    parser.add_argument("--power_model", "-pm", type=Path)
    

    #for split data
    parser.add_argument("--processStage1Data", "-pd", action="store_true")


    parser.add_argument("--customsplit","-c", action="store_true")
    parser.add_argument("--nonSplitData", "-d",type=Path)
    parser.add_argument("--split_randomseed","-rd", type=int, default=30)
    parser.add_argument("--n_occur", "-n", type=int, default=4)
    parser.add_argument("--sharedThroughputData", "-sd", type=Path)

    #stage2
    parser.add_argument("--processStage2Data", "-pd2", action="store_true")
    parser.add_argument("--sharedPowerData", "-power", type=Path)
    parser.add_argument("--sharedDurationData", "-duration", type=Path)
    parser.add_argument("--sharedEnergyData", "-energy", type=Path)
    parser.add_argument("--label_policy", "-lp", type=str, choices=["max-min", "maxthroughput-powercap", "separate_throughputpower_regression", "power_regression", "min-max", "freq-scale"], default="maxthroughput-powercap")


    #data
    #train file "output/dataset/training_set.csv"
    parser.add_argument("--baseline_file", "-bf", type=Path)
    parser.add_argument("--train_file", type=Path)
    #test file,="output/dataset/testing_set.csv"
    parser.add_argument("--test_file", type=Path)
    parser.add_argument("--correlation","-corr", type=float, default=0.2)
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--custom_col_exclude', '-excol', type=list_of_strings, default=[])

    #train
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--modeltype", "-mt", type=str, choices=["hotcloud", "KACE", "AutoML", "RF", "NN", "threadclass", "threadregression", "linear", "extratrees"])
    #for tree regularizatio tests...
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees (-1 for None)")
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--bootstrap", type=bool, default=False)
    parser.add_argument("--max_features", type=str, default=None)
    #baselines
    parser.add_argument("--rulebase", "-rb", action="store_true")

    #noHPworkloads - get mse/r2 for all workloads in testing set without specifying HPworkload
    parser.add_argument("--predAllacc", "-ga", action="store_true")

    #multuinstance
    parser.add_argument("--n_combination", "-comb", type=int, default=2)
    parser.add_argument("--pred_multiinstance", "-predmulti", action="store_true")

    #debug
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()
    # Convert max_depth: -1 means None
    print(f"args max_depth={args.max_depth}")
    args.max_depth = None if args.max_depth == -1 else args.max_depth
    args.bootstrap = True if args.bootstrap == "True" else False
    print(f"args max_depth after reading-1={args.max_depth}")
    if args.max_features == "None":
        args.max_features = 1.0

    # Logging Configuration
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(levelname)s [%(filename)s - %(funcName)s - line:%(lineno)d] - %(message)s',
        level=log_level
    )

    # Optional: Get a logger for main
    logger = logging.getLogger(__name__)
    logger.info("Starting the application.")


    

    #process stage1Data from mergeing kernels, shareddata, and baselines
    if args.processStage1Data:
        stage1Data = getstage1Data(args) 
        exit(0)
    if args.processStage2Data:
        stage2Data = getstage2Data(args)
        exit(0)
#split data if provided
    if args.customsplit:
        trainset, testset = train_test_split_with_counts(data=Path(args.nonSplitData),  
                                    train_outname=args.train_file, 
                                    test_outname=args.test_file, n_workload_counts=args.n_occur, random_seed=args.split_randomseed, standard_split=True)
        
    
        exit(0)
    #train, testfile should be provided together to delete low-correlation columns
    excluded_cols = []
    #train with provided train_file
    if args.train_file and args.train:
        print(f"process training data at {args.train_file}")
        train_dataloader = Dataloader(args, 
                                    baselineData="./tests/mps/analysis/0122_baseline_metrics.csv",
                                    kernelData="",
                                    shareThroughputData="",
                                    sharePowerData="",
                                    shareDurationData="",
                                    shareEnergyData="")
        X_train, X_test, y_train , y_test , test_workloads, columns_excluded =  train_dataloader.aggregate_and_split_train_test(data=args.train_file,  
                                                    train_postprocess_data="", #not used
                                                    test_workload="",
                                                    mode="train",
                                                    target=args.label_policy,
                                                    correlation=args.correlation,
                                                    n_combination=args.n_combination)
        #need excluded columns to pass to predictor for process test file...
        trainer = Trainer(args)
        modelpath = trainer.train(modeltype=args.modeltype,
                    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                )  
        print(f"model saved to {modelpath}")
        print(f"training file processed. excluded cols={excluded_cols}")
        exit(0)
    
    print("in main...")
    if args.train and not args.train_file:  
        model, excluded_cols = train(args) 
        print("training done. exit...")
        
        exit(0)
    
    if not args.train :
        testfile = Path(args.test_file) if args.test_file else  Path(args.nonSplitData)

        #predict - need to have sharedThroughputData
        predict(args, testfile, args.model,  excluded_cols)


    
    """
    #process stage1Data
    if args.processStage1Data:
        stage1Data = getstage1Data(args) 
        
    if args.customsplit:
        
        trainset, testset = train_test_split_with_counts(data=Path(args.stage1Data),  
                                     train_outname=args.training_file, 
                                     test_outname=args.testing_file, n_workload_counts=3)
        stage1Data = testset
            
    else:
        #load stage1Data from existed file
        stage1Data = Path(args.stage1Data)

    
    predict(args, model)

    
  """
        
    
    #HPworkloads = ["bert-base-cased_batch2-train","vit-base-patch16-224_batch2-inf", "vit_h_14_batch16-train", "wav2vec2-base-960h_batch8-inf", "bert-base-cased_batch2-inf"]
    
    #for HPworkload in HPworkloads:
    #    predict(HPworkload, stage1Data)
    
    #predict(args, 
    #        targetMPS=args.targetMPS)
    