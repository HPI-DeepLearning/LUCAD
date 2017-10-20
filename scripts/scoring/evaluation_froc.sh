#!/usr/bin/env bash

trap 'exit 130' INT

function usage_and_exit() {
    echo "usage: ./evaluation.sh output_folder [output_folder...]"
    exit 1
}

function check_script() {
    if [ ! -f "${1}" ]; then
        echo "could not find script: ${1}"
        echo ""
        usage_and_exit
    else
        echo "found: ${1}"
    fi
}

function get_script() {
    local result=${2}
    local path="${git_root}/${1}"
    local abs_path=$(realpath "${path}")
    check_script "${abs_path}"
    eval ${result}="'${abs_path}'"
}

script=$(readlink -f "$0")
script_dir=$(dirname "${script}")
git_root=$(readlink -f "${script_dir}")

while [ ! -f "${git_root}/README.md" ]; do
    git_root=$(readlink -f "${git_root}/..")
done

get_script "evaluation/noduleCADEvaluationLUNA16.py" evaluation_script

for output_folder in "$@"; do
    if [ ! -d "${output_folder}" ]; then
        echo "${output_folder} is not a directory."
        usage_and_exit
    fi
done

for output_folder in "$@"; do
    echo "Processing ${output_folder}..."

    results_file="${output_folder}/concat.csv"

    ANN_ROOT="${git_root}/evaluation/annotations"

    mkdir -p "${output_folder}/CADEvaluation"

    echo "Starting evaluation, this can take a while..."

    python ${evaluation_script} "${ANN_ROOT}/annotations.csv" "${ANN_ROOT}/annotations_excluded.csv" "${ANN_ROOT}/seriesuids.csv" "${results_file}" "${output_folder}/CADEvaluation" &> "${output_folder}/CADEvaluation/evaluation.log"

    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "error with evaluation script command:"
        echo "   python ${evaluation_script} \"${ANN_ROOT}/annotations.csv\" \"${ANN_ROOT}/annotations_excluded.csv\" \"${ANN_ROOT}/seriesuids.csv\" \"${output_folder}/concat.csv\" \"${output_folder}/CADEvaluation\""
        echo "check log file: ${output_folder}/CADEvaluation/evaluation.log"
        exit 1
    fi

    echo "Finished! Results can be found in ${output_folder}/CADEvaluation."
done
