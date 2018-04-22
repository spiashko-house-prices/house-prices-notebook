#!/usr/bin/env bash

docker run -it --rm --name house-prices-notebook -e="PORT=8000" -e="PASSWORD=" -p 8888:8000 \
house-prices-notebook
