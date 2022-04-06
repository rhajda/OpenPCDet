FROM openpcdet-base:latest

RUN apt-get install libgl1-mesa-glx vim -y

RUN pip install open3d

WORKDIR /home
COPY . .

RUN python setup.py develop

ENV PYTHONPATH=/home
ENV XAUTHORITY=/home/.Xauthority
