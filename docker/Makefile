DOCKER=docker
TAG=latest

all: eurus-base eurus-server

eurus-base: Dockerfile-base
    $(DOCKER) build -t hudl/eurus-base:$(TAG) -f Dockerfile-base ..

eurus-server: ../eurus
	$(DOCKER) build -t hudl/eurus-server:$(TAG) -f Dockerfile-server ..