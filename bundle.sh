#!/bin/bash

# put all the subdirectories into their own tar.gz files
for dir in */ ; do
    dir_name="${dir%/}"
    tar -czf "${dir_name}.tar.gz" "$dir_name" &
    
    # Limit to 4 parallel jobs
    while [ $(jobs -r | wc -l) -ge 4 ]; do
        sleep 0.1
    done
done

# Wait for remaining jobs to complete
wait