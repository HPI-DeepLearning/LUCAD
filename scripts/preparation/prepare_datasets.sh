#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "usage: ./prepare_datasets.sh data_root"
    exit 1
fi

DATA_ROOT=$1

OUTPUT_DIR="${DATA_ROOT}/v2_none"
mkdir -p "${OUTPUT_DIR}"
python scripts/preparation/prepare_dataset.py \
    --storage memmap \
    --augmentation none \
    --v2 \
    ${DATA_ROOT}/original \
    ${OUTPUT_DIR} \
    > ${OUTPUT_DIR}/preparation.log


OUTPUT_DIR="${DATA_ROOT}/v2_dice_memmap_shuffled"
mkdir -p "${OUTPUT_DIR}"
python scripts/preparation/prepare_dataset.py \
    --storage memmap \
    --augmentation dice \
    --shuffle \
    --v2 \
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
