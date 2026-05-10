import subprocess
import re
import os
import yaml

def parse_gpushare_info():
    # Run the kubectl inspect gpushare command and capture the output
    result = subprocess.run("kubectl inspect gpushare", shell=True, capture_output=True, text=True)

    # Extract information using regular expressions
    gpus_info = []
    pattern = r"(\w+)\s+(\d+\.\d+\.\d+\.\d+)\s+(\d+)\/(\d+)\s+(\d+)\/(\d+)"
    matches = re.findall(pattern, result.stdout)

    for match in matches:
        gpu_info = {
            "Name": match[0],
            "IPAddress": match[1],
            "GPU0Mem_Allocated": int(match[2]),
            "GPU0Mem_Total": int(match[3]),
            "NodeMem_Allocated": int(match[4]),
            "NodeMem_Total": int(match[5])
        }
        gpus_info.append(gpu_info)
        
    return gpus_info


"""
extract GPU_MEMORY from profiled k8s job yaml
"""
def extract_gpu_memory(filename_be):
    with open(filename_be, 'r') as file:
        yaml_content = yaml.safe_load(file)

    gpu_memory = yaml_content.get("spec", {}).get("containers", [{}])[0].get("env", [{}])
    for env_var in gpu_memory:
        if env_var.get("name") == "GPU_MEMORY":
            return int(env_var.get("value", 0))

    return 0  # Return 0 if GPU_MEMORY is not found in the YAML conten