nvidia-docker run --rm --name trtserver -p 8000:8000 -p 8001:8001  -v `pwd`:/models nvcr.io/nvidia/tensorrtserver:19.03-py3 trtserver --model-store=/models
