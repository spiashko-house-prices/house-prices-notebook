#!/usr/bin/env bash

docker run -it --rm --name house-prices-notebook -p 8888:8888 \
house-prices-back
