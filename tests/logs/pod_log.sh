#!/bin/bash

# Set the namespace
namespace="default"

# Get the list of pod names in the specified namespace
pod_names=$(kubectl get pods --namespace="$namespace" -o jsonpath='{.items[*].metadata.name}')

# Loop through each pod and fetch its logs
for pod in $pod_names; do
    echo "Fetching logs for pod: $pod"
    kubectl logs "$pod" --namespace="$namespace" > "${pod}_logs.txt"
    echo "Logs from pod $pod have been saved to ${pod}_logs.txt"
done
