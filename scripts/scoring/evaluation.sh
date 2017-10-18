#!/usr/bin/env bash

function usage_and_exit() {
    echo "usage: ./evaluation.sh output_folder"
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

if [ "$#" -ne 1 ]; then
    usage_and_exit
fi

OUTPUT_FOLDER="$1"

SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "${SCRIPT}")
GIT_ROOT=$(readlink -f "${SCRIPT_DIR}")

while [ ! -f "$GIT_ROOT/README.md" ]; do
    GIT_ROOT=$(readlink -f "${GIT_ROOT}/..")
done

EVALUATION_SCRIPT_PATH="evaluation/noduleCADEvaluationLUNA16.py"
EVALUATION_SCRIPT="${GIT_ROOT}/${EVALUATION_SCRIPT_PATH}"

check_script "${EVALUATION_SCRIPT}"

CONCAT_SCRIPT_PATH="scripts/scoring/concat.sh"
CONCAT_SCRIPT="${GIT_ROOT}/${CONCAT_SCRIPT_PATH}"

check_script "${CONCAT_SCRIPT}"

${CONCAT_SCRIPT} "${OUTPUT_FOLDER}" "concat.csv"

RESULTS_FILE="${OUTPUT_FOLDER}/concat.csv"

NUM_LINES=$(wc -l "${RESULTS_FILE}")

if [[ "${NUM_LINES:0:6}" == "754976" ]]; then
    echo "Concatenation finished: result file ${RESULTS_FILE} has correct number of lines."
else
    echo "the result file ${RESULTS_FILE} has a wrong number of lines (${NUM_LINES:0:6} instead of 754976), maybe some output files are missing?"
    exit 1
fi

ANN_ROOT="${GIT_ROOT}/evaluation/annotations"

mkdir -p "${OUTPUT_FOLDER}/CADEvaluation"

echo "Starting evaluation, this can take a while..."

python ${EVALUATION_SCRIPT} "${ANN_ROOT}/annotations.csv" "${ANN_ROOT}/annotations_excluded.csv" "${ANN_ROOT}/seriesuids.csv" "${RESULTS_FILE}" "${OUTPUT_FOLDER}/CADEvaluation" &> "${OUTPUT_FOLDER}/CADEvaluation/evaluation.log"

STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "error with evaluation script command:"
    echo "   python ${EVALUATION_SCRIPT} \"${ANN_ROOT}/annotations.csv\" \"${ANN_ROOT}/annotations_excluded.csv\" \"${ANN_ROOT}/seriesuids.csv\" \"${OUTPUT_FOLDER}/concat.csv\" \"${OUTPUT_FOLDER}/CADEvaluation\""
    echo "check log file: ${OUTPUT_FOLDER}/CADEvaluation/evaluation.log"
    exit 1
fi

echo "Finished! Results can be found in ${OUTPUT_FOLDER}/CADEvaluation."
