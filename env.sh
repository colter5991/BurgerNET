#!/bin/bash
docker build -t burgernet-build .
docker run -it \
  -v $PWD:$PWD \
  -v /etc/passwd:/etc/passwd:ro \
  -u $(id -u) \
  -w $PWD \
  --net host \
  --gpus all \
  burgernet-build
  #nvidia/cuda:11.2.0-devel /bin/bash
