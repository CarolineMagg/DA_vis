FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

ENV DEBIAN_FRONTEND noninteractive

# general libs
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libpng16-16 \
	libzmq3-dev \
	pkg-config \
	libsm6 \
	libxext6 \
	graphviz \
	python3.6-tk \
	git \
	vim \
	apt-utils \
	libxtst-dev libxext-dev libxrender-dev libfreetype6-dev libfontconfig1 libgtk2.0-0 libxslt1.1 libxxf86vm1 \
	wget \
	curl \
	software-properties-common \
	make \
	gcc \
	g++ \
	bzip2 \
	hdf5-tools \
	unzip \
	gfortran \
	curl \
	sudo \
	libx11-xcb1 libasound2 x11-apps libice6 libsm6 libxaw7 libxft2 libxmu6 libxpm4 libxt6 x11-apps xbitmaps \
	mesa-utils libgl1-mesa-glx libxcb-xinerama0 \
	python3-pyqt5 
	
RUN apt-get install libpng-dev libx11-dev libxt-dev libgl1-mesa-dev libglu1-mesa-dev libfontconfig-dev libxrender-dev libncurses5-dev -yq

# python-specific libs	
RUN pip3.6 --no-cache-dir install \
	pandas \
	numpy \
	pydicom \	
	matplotlib \
	scikit-image \
	scikit-learn \
	scipy \
	opencv-python==4.1.0.25 \
	pydot \
	natsort \
	nibabel \
	albumentations \
	tensorflow-addons \
	MedPy==0.4.0 
	
# python-specific libs	visualization
RUN pip3.6 --no-cache-dir install \
	plotly==5.3.1 \
	dash \
	jupyter-dash \
	pyradiomics \
	dash-bootstrap-components \
	kaleido

# pycharm
RUN mkdir /pch 
COPY installer.tgz /pch/
RUN cd /pch && tar --strip-components=1 -xzf /pch/installer.tgz && rm /pch/installer.tgz
ENV PATH /pch/bin:$PATH

# add user and workdir
RUN adduser --disabled-password --gecos '' caroline
RUN usermod -aG sudo caroline
USER caroline
WORKDIR /home/caroline/
#WORKDIR /tf/

# jupyter notebook
RUN jupyter notebook --generate-config
COPY jupyter_notebook_config.py /home/caroline/.jupyter/

WORKDIR /home/caroline/
EXPOSE 8050
