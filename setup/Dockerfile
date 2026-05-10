#upgrade current ml_tasks image to latest version. please refer to ~/Dockerfile for original Dockerfile
#TODO 05312025 - this script should not run since it cannot run autoregressive other than gpt2-xl torchvision messed up latest torch dependency. need to figure a way to solve both torch and torchvision dependency issues(use latest nvidia docker ?) 
FROM nba556677/ml_tasks:latest

# Update Python libraries. install with cuda11.8
RUN pip uninstall -y torch torchvision transformers
RUN pip cache purge
RUN pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install transformers==4.52.4
# You might need to set the entrypoint or cmd if the base image doesn't have it
# or if you need to override it. For your script, this might not be necessary
# as your script specifies the command to run inside the container.


#push to Docker Hub
#docker build -t nba556677/ml_tasks:latest .
#docker push nba556677/ml_tasks:latest

