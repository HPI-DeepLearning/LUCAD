#!/usr/bin/env bash

function usage_and_exit() {
    echo "usage: ./concat.sh output_folder output_file"
    echo "execute in a folder with output[0-9].csv"
    exit 1
}

function check_file() {
    if [ ! -f "${1}" ]; then
        echo "could not find file: ${1}"
        echo ""
        usage_and_exit
    fi
}

function copy_header() {
    check_file "${1}"
    head -n 1 "${1}" > "${2}"
}

function copy_body() {
    check_file "${1}"
    tail -n +2 "${1}" >> "${2}"
}


if [ "$#" -ne 2 ]; then
    usage_and_exit
fi

OUTPUT_FOLDER="$1"
OUTPUT_FILE="$2"

copy_header "${OUTPUT_FOLDER}/output0.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"

copy_body "${OUTPUT_FOLDER}/output0.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output1.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output2.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output3.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output4.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output5.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output6.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output7.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output8.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
copy_body "${OUTPUT_FOLDER}/output9.csv" "${OUTPUT_FOLDER}/${OUTPUT_FILE}"
