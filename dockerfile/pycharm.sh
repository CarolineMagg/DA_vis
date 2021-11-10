#!/bin/bash
xhost +

docker run --gpus all -d -t --rm --name pycharm \
	    -v /home/caroline/Documents/DiplomaThesis:/tf/workdir \
	    -v /home/caroline:/home/caroline \
	    -v /tmp/.X11-unix:/tmp/.X11-unix \
	    -e DISPLAY=$DISPLAY \
	    --user=caroline \
	    -p 8060:8060 \
	    --entrypoint=/pch/bin/pycharm.sh \
	    python:2.00
