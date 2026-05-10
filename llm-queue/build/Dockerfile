FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /app
COPY ./FlexGen-main .
RUN pip install flexgen pandas sentence-transformers rich pyarrow
WORKDIR /app/flexgen/apps/data_wrangle
RUN echo "$PWD" && ls
SHELL ["/bin/bash", "-c"]
#RUN bash ./apps/data_wrangle/install.sh
#RUN ["chmod", "+x", "./apps/data_wrangle/install.sh"]
#ENTRYPOINT [ "bash", "./apps/data_wrangle/install.sh" ] 
