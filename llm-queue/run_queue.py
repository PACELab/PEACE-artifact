import os
import argparse
import subprocess
import heapq
import time
from utils.util import parse_gpushare_info, extract_gpu_memory
import yaml

IPTable = {"node5" : "192.168.1.138",
           "node6" : "192.168.1.40"}

def create_job_queue(input_dir, num_jobs):
    job_queue = []
    idx = 0
    for filename in sorted(os.listdir(input_dir)):
        if idx == num_jobs:
            break
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            heapq.heappush(job_queue, filename)
        idx += 1
    return job_queue

def check_jobs_running(namespace, node):
    # Use kubectl command to check if any jobs are running in the specified namespace
    cmd = f"kubectl get pods -n {namespace} --field-selector spec.nodeName={node} | grep -E 'Running|ContainerCreating'"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"check_jobs_results stdout of {namespace} in {node} = {result.stdout.splitlines()}")
    print(f"num of jobs in {node} = {len(result.stdout.splitlines())}")
    return len(result.stdout.splitlines())

def schedule_job(filename, node):
    # Read the Job YAML file
    with open(filename, 'r') as file:
        job_yaml = yaml.safe_load(file)

    # Add the node field to the template spec
    node_name = node
    job_yaml['spec']['nodeSelector']['kubernetes.io/hostname']= node_name

    # Write the modified YAML back to a file
    with open('modified_job.yaml', 'w') as file:
        yaml.dump(job_yaml, file)

    # Use kubectl create to execute the modified Job
    subprocess.run(f"kubectl create -f modified_job.yaml", shell=True)

def schedule_tmp(filename):
    # Read the Job YAML file
    with open(filename, 'r') as file:
        job_yaml = yaml.safe_load(file)
    #print(filename)
    node = "node5" 
    #print(job_yaml["metadata"]["name"])
    
    if int(job_yaml["metadata"]["name"][7:8]) % 2 == 0:
    #if int(filename[8:9]) 
        node = "node6"
    # Add the node field to the template spec
    node_name = node
    job_yaml['spec']['nodeSelector']['kubernetes.io/hostname']= node_name

    # Write the modified YAML back to a file
    with open('modified_job.yaml', 'w') as file:
        yaml.dump(job_yaml, file)

    # Use kubectl create to execute the modified Job
    subprocess.run(f"kubectl create -f modified_job.yaml", shell=True)


def run_chatbot(args):
    CHATBOT_MEM = args.chatbot_mem
    prev_node = None
    runningJobs = {n:0 for n in args.available_nodes}
    #start sending inference requests
    #for node in args.available_nodes:
    #    subprocess.run(f"bash k8sJobs/chatbot/request.sh {args.chatbot_requests} {IPTable[node]} {args.chatbot_logdir} &", shell=True)
    
    #schedule be queue
    be_queue = create_job_queue(args.input_be, args.be_jobs)
    #node_list = args.available_nodes
    #run inference script
    while True:
        gpus_info = parse_gpushare_info()
        for node in args.available_nodes:
            
            # Check if BE jobs are running or completed
            be_jobs_running = check_jobs_running("be", node)
            #runningJobs[node] = be_jobs_running

            if be_jobs_running < args.be_num_colocate and be_queue:
            #if be_queue:
                # If no BE jobs are running and BE queue is not empty, create the top element
                gpus_info = parse_gpushare_info()
                #peek top
                top_be_job = be_queue[0]
                
                filename_be = os.path.join(args.input_be, top_be_job)
                requested_GPU_memory = extract_gpu_memory(filename_be)
                for gpu_info in gpus_info:
                    #any node that has sufficient memory could place 
                    if gpu_info['Name'] == node:
                        #print(gpu_info)
                        print(f"remained mem in {gpu_info['Name']} = {gpu_info['NodeMem_Total'] - gpu_info['NodeMem_Allocated'] - CHATBOT_MEM}MB, be_job requested {requested_GPU_memory} MB")
                        if gpu_info["NodeMem_Total"] - gpu_info["NodeMem_Allocated"] - CHATBOT_MEM > requested_GPU_memory:
                            # If there is sufficient memory available, create the BE job
                            print(f"starting be job {top_be_job}")
                            _ = heapq.heappop(be_queue)
                            # Use kubectl create to execute the modified Job
                            #schedule_job(filename_be, node)
                            schedule_tmp(filename_be)

                            #nodelist - place used 
                            break

        # Sleep for some time before checking again
        time.sleep(5)  # Sleep for 5 seconds
    
    


def run(args):
    # Create queues for LS and BE jobs
    ls_queue = create_job_queue(args.input_ls, args.ls_jobs)
    be_queue = create_job_queue(args.input_be, args.be_jobs)

    while True:
        gpus_info = parse_gpushare_info()
        for node in args.available_nodes:
            
            # Check if LS jobs are running or completed
            ls_jobs_running = check_jobs_running("ls", node)

            if not ls_jobs_running and ls_queue:
                # If no LS jobs are running and LS queue is not empty, create the top element
                #peek top
                top_ls_job = ls_queue[0]
                
                filename_ls = os.path.join(args.input_ls, top_ls_job)
                requested_GPU_memory = extract_gpu_memory(filename_ls)
                for gpu_info in gpus_info:
                    #any node that has sufficient memory could place 
                    if gpu_info['Name'] == node:
                        print(f"remained mem in {gpu_info['Name']} = {gpu_info['NodeMem_Total'] - gpu_info['NodeMem_Allocated']}MB, ls_job requested {requested_GPU_memory} MB")
                        if gpu_info["NodeMem_Total"] - gpu_info["NodeMem_Allocated"] > requested_GPU_memory:
                            # If there is sufficient memory available, create the BE job
                            print(f"starting ls job {top_ls_job}")
                            _ = heapq.heappop(ls_queue)
                            schedule_job(filename_ls, node)
                            break
            #sleep in case gpu share is not updated yet 
            time.sleep(2)
            # Check if BE jobs are running or completed
            be_jobs_running = check_jobs_running("be", node)
                
            if not be_jobs_running and be_queue:
                # If no BE jobs are running and BE queue is not empty, create the top element
                gpus_info = parse_gpushare_info()
                #peek top
                top_be_job = be_queue[0]
                
                filename_be = os.path.join(args.input_be, top_be_job)
                requested_GPU_memory = extract_gpu_memory(filename_be)
                for gpu_info in gpus_info:
                    #any node that has sufficient memory could place 
                    if gpu_info['Name'] == node:
                        #print(gpu_info)
                        print(f"remained mem in {gpu_info['Name']} = {gpu_info['NodeMem_Total'] - gpu_info['NodeMem_Allocated']}MB, be_job requested {requested_GPU_memory} MB")
                        if gpu_info["NodeMem_Total"] - gpu_info["NodeMem_Allocated"] > requested_GPU_memory:
                            # If there is sufficient memory available, create the BE job
                            print(f"starting be job {top_be_job}")
                            _ = heapq.heappop(be_queue)
                            schedule_job(filename_be, node)
                            break

        # Sleep for some time before checking again
        time.sleep(10)  # Sleep for 10 seconds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ls", type=str, default="k8sJobs/ls", help="k8s job dir for latency-sensitive tasks")
    parser.add_argument("--input_be", type=str, default="k8sJobs/be", help="k8s job dir for best-effort tasks")
    parser.add_argument("--available_nodes", nargs='+' ,help='<Required> node name', required=True)
    parser.add_argument("--ls_jobs", type=int, default=6)
    parser.add_argument("--be_jobs", type=int, default=6)
    parser.add_argument("--be_num_colocate", type=int, default=1)
    
    #chatbot args
    parser.add_argument("--chatbot", action="store_true", help="simulates interactive chatbot for LS jobs. LS queue only includes requests")
    parser.add_argument("--chatbot_mem", type=int, default=18950)
    parser.add_argument("--chatbot_logdir", type=str, default="logs", help="chatbot logsdir")
    parser.add_argument("--chatbot_requests", type=str, help="script name for chatbot api call", required=True) 
    
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.chatbot:
        run_chatbot(args)
    else:
        run(args)
