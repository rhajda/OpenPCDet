
# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /home
WORKDIR /home
COPY . .

# Install TensorFlow 2.1.0, upgrade pip, and install required packages
RUN pip install tensorflow==2.1.0 \
    && pip install --upgrade pip \
    && pip install waymo-open-dataset-tf-2-1-0==1.2.0 \
    && pip install opencv-python matplotlib tqdm \
    && pip install open3d

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


