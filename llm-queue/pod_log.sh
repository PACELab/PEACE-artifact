#!/bin/bash

# Set the namespace
namespace=$1
output_dir=$2
# Check if both $1 and $2 are provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide two arguments: namespace and output_dir."
    exit 1
fi
mkdir -p $output_dir

# Get the list of pod names in the specified namespace
pod_names=$(kubectl get pods --namespace="$namespace" -o jsonpath='{.items[*].metadata.name}')

# Loop through each pod and fetch its logs
for pod in $pod_names; do
    echo "Fetching logs for pod: $pod"
    kubectl logs "$pod" --namespace="$namespace" > "${output_dir}/${pod}_logs.txt"
    kubectl get pod "$pod"  -n be -o json | jq -r '.status.conditions' > "${output_dir}/${pod}_timestamps.txt"
    echo "Logs from pod $pod have been saved to ${output_dir}/${pod}_logs.txt"
done
