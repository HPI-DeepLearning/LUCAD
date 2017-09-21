#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "usage: ./concat.sh output_file"
    echo "execute in a folder with output[0-9].csv"
    exit 1
fi

OUTPUT_FILE="$1"

head -n 1 output0.csv > ${OUTPUT_FILE}

tail -n +2 output0.csv >> ${OUTPUT_FILE}
tail -n +2 output1.csv >> ${OUTPUT_FILE}
tail -n +2 output2.csv >> ${OUTPUT_FILE}
tail -n +2 output3.csv >> ${OUTPUT_FILE}
tail -n +2 output4.csv >> ${OUTPUT_FILE}
tail -n +2 output5.csv >> ${OUTPUT_FILE}
tail -n +2 output6.csv >> ${OUTPUT_FILE}
tail -n +2 output7.csv >> ${OUTPUT_FILE}
tail -n +2 output8.csv >> ${OUTPUT_FILE}
tail -n +2 output9.csv >> ${OUTPUT_FILE}
