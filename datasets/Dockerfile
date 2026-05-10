
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /app
RUN pip install datasets
COPY load_dataset.py
# Specify the mounted volume
VOLUME /home/root/.cache/huggingface