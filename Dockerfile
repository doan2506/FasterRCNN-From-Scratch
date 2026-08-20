# Syntax: docker/dockerfile:1
# Pinned official PyTorch runtime used for both student development and grading.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-exam.txt /opt/object-detection-exam/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/object-detection-exam/requirements.txt \
    && python -c "import albumentations, cv2, matplotlib, numpy, PIL, scipy, tensorboard, torch, torchvision, yaml; assert torch.__version__.startswith('2.7.1'); assert torchvision.__version__.startswith('0.22.1')"

WORKDIR /workspace
CMD ["python", "--version"]
