FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    apt-get install -y libgl1-mesa-glx && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


RUN python --version && pip --version


RUN pip install --upgrade pip && \
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install \
        monai==1.4.0 \
        lightning==2.5.0 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        pyyaml==6.0.2 \
        matplotlib==3.9.0 \
        scipy==1.12.0 \
        SimpleITK==2.4.0 \
        opencv-python==4.10.0.82 \
        json5==0.10.0

COPY ./ ./

WORKDIR /
CMD ["python", "infer.py", "config_files/infer.yaml"]
