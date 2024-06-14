#!/bin/bash

for file in examples/*.py; do
    echo "Running $file"

    poetry run python $file
done
