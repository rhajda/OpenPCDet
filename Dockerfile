FROM openpcdet:latest

# RUN apt-get -o Dpkg::Options::="--force-overwrite" install nvidia-cuda-toolkit -y --fix-broken
RUN pip install setuptools==60.7.0
RUN apt-get install libgl1-mesa-glx vim -y

RUN pip install --upgrade pip

RUN pip install open3d --ignore-installed PyYAML
# RUN pip install mayavi

WORKDIR /home
ADD . . 

# RUN python setup.py develop 
# Better:
RUN pip install -e .

ENV PYTHONPATH=/home
ENV XAUTHORITY=/home/.Xauthority
