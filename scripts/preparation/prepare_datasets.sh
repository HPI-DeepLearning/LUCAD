#!/usr/bin/env bash

function usage_and_exit() {
    echo "usage: ./prepare_datasets.sh data_root configuration [source]"
    echo "    data_root     - path to data directories should contain 'original' folder with original data"
    echo "    configuration - [normal, fonova7, fonova100, test, kokA, kokB, xyA, xyB, xyC]"
    echo "    source        - [luna, tianchi], default: luna"
    exit 1
}

DATA_ROOT="${1}"
CONFIG="${2}"

BASE_DATA=""
if [ "$#" -eq 2 ]; then
    BASE_DATA="original"
fi

if [ "$#" -eq 3 ]; then
    if [ "${3}" == "luna" ]; then
        BASE_DATA="original"
    fi

    if [ "${3}" == "tianchi" ]; then
        BASE_DATA="tianchi-dataset"
    fi
fi

if [ "${BASE_DATA}" == "" ]; then
    usage_and_exit
fi

if [ "${BASE_DATA}" == "luna" ]; then
    DATA_PREFIX="original"
fi

if [ "${BASE_DATA}" == "tianchi-dataset" ]; then
    DATA_PREFIX="tianchi_${CONFIG}"
fi

ORIGINAL_DATA="${DATA_ROOT}/${BASE_DATA}"

OPTIONS=""
if [ "${CONFIG}" == "test" ]; then
    OPTIONS="--storage memmap --augmentation none"
fi

if [ "${CONFIG}" == "normal" ]; then
    OPTIONS="--storage memmap --augmentation dice --shuffle"
fi

if [ "${CONFIG}" == "fonova7" ]; then
    OPTIONS="--storage memmap --augmentation fonova --shuffle --voxelsize 0.5556"
fi

if [ "${CONFIG}" == "fonova100" ]; then
    OPTIONS="--storage memmap --augmentation fonova --shuffle --voxelsize 0.5556 --factor 100"
fi

if [ "${CONFIG}" == "kokA" ]; then
    OPTIONS="--storage memmap --augmentation kok --shuffle --factor 42 --ratio 5"
fi

if [ "${CONFIG}" == "kokB" ]; then
    OPTIONS="--storage memmap --augmentation kok --shuffle --ratio 5"
fi

if [ "${CONFIG}" == "xyA" ]; then
    OPTIONS="--storage memmap --augmentation xy --shuffle --factor 100 --ratio 5"
fi

if [ "${CONFIG}" == "xyB" ]; then
    OPTIONS="--storage memmap --augmentation xy --shuffle --factor 50 --ratio 10"
fi

if [ "${CONFIG}" == "xyC" ]; then
    OPTIONS="--storage memmap --augmentation xy --shuffle --factor 25 --ratio 20"
fi

if [ "${OPTIONS}" == "" ]; then
    usage_and_exit
fi

OUTPUT_DIR="${DATA_ROOT}/${DATA_PREFIX}"
mkdir -p "${OUTPUT_DIR}"
python scripts/preparation/prepare_dataset.py ${OPTIONS} \
    "${ORIGINAL_DATA}" \
    "${OUTPUT_DIR}" \
    > "${OUTPUT_DIR}/preparation.log"

##### ##### old versions below ##### #####

# for training and validation:
#OUTPUT_DIR="${DATA_ROOT}/dice_memmap_shuffled"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage memmap \
#    --augmentation dice \
#    --shuffle \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
#
#OUTPUT_DIR="${DATA_ROOT}/dice_raw_shuffled"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage raw \
#    --augmentation dice \
#    --shuffle \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
#
#OUTPUT_DIR="${DATA_ROOT}/nozflip_memmap_shuffled"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage memmap \
#    --augmentation nozflip \
#    --shuffle \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
#
#OUTPUT_DIR="${DATA_ROOT}/nozflip_raw_shuffled"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage raw \
#    --augmentation nozflip \
#    --shuffle \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
#
#
## for testing:
#OUTPUT_DIR="${DATA_ROOT}/none_memmap"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage memmap \
#    --augmentation none \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
#    # == non-augmented
#
#OUTPUT_DIR="${DATA_ROOT}/none_raw"
#mkdir -p "${OUTPUT_DIR}"
#python scripts/preparation/prepare_dataset.py \
#    --storage raw \
#    --augmentation none \
#    ${DATA_ROOT}/original \
#    ${OUTPUT_DIR} \
#    > ${OUTPUT_DIR}/preparation.log
