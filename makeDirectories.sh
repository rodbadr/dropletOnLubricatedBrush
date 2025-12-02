#!/bin/bash

# Loop through all items in the current directory
for dir in */ ; do
    # Check if it's a directory
    if [ -d "$dir" ]; then
        mkdir -p "$dir/initialConditions"
        mkdir -p "$dir/trajectories"
    fi
done
