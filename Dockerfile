<<<<<<< HEAD
# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.234.0/containers/python-3/.devcontainer/base.Dockerfile

FROM openpcdet-base:latest
=======
FROM openpcdet-base113:latest
>>>>>>> origin/determinism

# RUN apt-get -o Dpkg::Options::="--force-overwrite" install nvidia-cuda-toolkit -y --fix-broken
RUN pip install --upgrade pip
RUN pip install setuptools==60.7.0

RUN apt-get update
RUN apt-get install libgl1-mesa-glx vim -y

# TO fix: 
# >>> import open3d
# Traceback (most recent call last):
# OSError: libGL.so.1: cannot open shared object file: No such file 
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install open3d --ignore-installed PyYAML
# RUN pip install mayavi

WORKDIR /home
RUN git config --global --add safe.directory /home


RUN mkdir data
ADD . . 

# RUN python setup.py develop
RUN pip install -e .

ENV PYTHONPATH=/home
ENV XAUTHORITY=/home/.Xauthority
<<<<<<< HEAD


# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
=======
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
>>>>>>> origin/determinism
