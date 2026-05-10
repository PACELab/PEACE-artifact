import subprocess
from run_queue import check_jobs_running

import os
import yaml


# Read the Job YAML file
with open('k8sJobs/ls/llm.yaml', 'r') as file:
    job_yaml = yaml.safe_load(file)

# Add the node field to the template spec
node_name = 'node5'
job_yaml['spec']['nodeSelector']['kubernetes.io/hostname']= node_name

# Write the modified YAML back to a file
with open('modified_job.yaml', 'w') as file:
    yaml.dump(job_yaml, file)

# Use kubectl create to execute the modified Job
subprocess.run(f"kubectl create -f modified_job.yaml", shell=True)