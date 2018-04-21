#!/usr/bin/env bash

docker run -it --rm --name house-prices-notebook -e="PORT=8888" -p 8889:8888 \
house-prices-notebook
