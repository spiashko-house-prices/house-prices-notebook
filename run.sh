#!/usr/bin/env bash

docker run -it --rm --name house-prices-notebook -e="PORT=8888" -e="PASSWORD=" -p 8888:8888 \
house-prices-notebook
