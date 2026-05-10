FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /app
RUN pip install datasets transformers numpy evaluate accelerate scikit-learn tqdm pathlib wandb matplotlib pandas termcolor

#
ARG API_KEY
# Set environment variables
ENV WANDB_API_KEY=${API_KEY} \
    WANDB_SILENT=true
# Run the script to perform the copy operation
COPY template ./template
COPY utils ./utils
COPY workloads ./workloads
COPY models ./models
WORKDIR /app/template

RUN wget http://ftp.gnu.org/gnu/libc/glibc-2.29.tar.gz && \
    tar -zxvf glibc-2.29.tar.gz && \
    cd glibc-2.29 && \
    mkdir build && cd build && \
    ../configure --prefix=/opt/glibc-2.29 && \
    make -j$(nproc) && \
    make install && \
    rm -rf /glibc-2.29.tar.gz /glibc-2.29

ENV LD_LIBRARY_PATH=/opt/glibc-2.29/lib:$LD_LIBRARY_PATH

#TODO Run Dockerfile and see if the command runs
#docker run [IMAGE] python [train_file] --n_epoch 20 --profile --profile_all_estimation
