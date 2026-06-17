#!/bin/bash

DEST="/home/tom/telugu_data/sample_set/val/10"

cd /home/tom/telugu_data/TeluguSeg/val/10;

for num in "$@"; do
	echo "$num running...";
    cp -r "./$num" "$DEST"
done
