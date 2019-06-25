#!/usr/bin/env bash

DOCKER_IMAGE_NAME=tag_translation_research

DATA_DIR=$(pwd)/data

mkdir -p $DATA_DIR/acousticbrainz

cd docker && docker build . -t $DOCKER_IMAGE_NAME && cd -;

docker run -v $(pwd):/srv/workspace/tag_translation_research $DOCKER_IMAGE_NAME:latest make plots;
