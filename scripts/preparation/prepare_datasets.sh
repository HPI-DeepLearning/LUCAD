#!/usr/bin/env bash

function usage_and_exit() {
    echo "usage: ./prepare_datasets.sh data_root configuration"
    echo "    data_root     - path to data direcotires should contain 'original' folder with original data"
    echo "    configuration - [normal, fonova, test]"
    exit 1
}

if [ "$#" -ne 2 ]; then
    usage_and_exit
fi

DATA_ROOT=$1
CONFIG=$2
DATA_PREFIX="v2_${CONFIG}"

OPTIONS=""
if [ "${CONFIG}" == "test" ]; then
    OPTIONS="--storage memmap --augmentation none"
fi

if [ "${CONFIG}" == "normal" ]; then
    OPTIONS="--storage memmap --augmentation dice --shuffle"
fi

if [ "${CONFIG}" == "fonova" ]; then
    OPTIONS="--storage memmap --augmentation fonova --shuffle --voxelsize 0.5556"
fi

if [ "${OPTIONS}" == "" ]; then
    usage_and_exit
fi

OUTPUT_DIR="${DATA_ROOT}/${DATA_PREFIX}"
mkdir -p "${OUTPUT_DIR}"
python scripts/preparation/prepare_dataset.py ${OPTIONS} \
    ${DATA_ROOT}/original \
    ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/preparation.log

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
