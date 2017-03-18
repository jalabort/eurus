DOCKER=docker
TAG=latest

all: eurus-base eurus-dev eurus-server

eurus-base:
    sed -e 's/:latest/:$(TAG)/' Dockerfile-base > tagged.dkf
    $(DOCKER) build -t hudl/eurus-base:$(TAG) -f Dockerfile-base ..
    rm tagged.dkf

eurus-dev:
    sed -e 's/:latest/:$(TAG)/' Dockerfile-dev > tagged.dkf
    $(DOCKER) build -t hudl/eurus-dev:$(TAG) -f Dockerfile-dev ..
    rm tagged.dkf

eurus-server:
    sed -e 's/:latest/:$(TAG)/' Dockerfile-server> tagged.dkf
	$(DOCKER) build -t hudl/eurus-server:$(TAG) -f Dockerfile-server ..
    rm tagged.dkf
