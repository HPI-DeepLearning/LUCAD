#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
    echo "usage: ./prepare_datasets.sh data_root"
    exit 1
fi

DATA_ROOT=$1

# for training and validation:
python scripts/preparation/prepare_dataset.py \
    --storage memmap \
    --augmentation dice \
    --shuffled \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/dice_memmap_shuffled \
    > ${DATA_ROOT}/dice_memmap_shuffled/preparation.log

python scripts/preparation/prepare_dataset.py \
    --storage memmap \
    --augmentation nozflip \
    --shuffled \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/nozflip_memmap_shuffled \
    > ${DATA_ROOT}/nozflip_memmap_shuffled/preparation.log

python scripts/preparation/prepare_dataset.py \
    --storage raw \
    --augmentation dice \
    --shuffled \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/dice_raw_shuffled \
    > ${DATA_ROOT}/dice_raw_shuffled/preparation.log

python scripts/preparation/prepare_dataset.py \
    --storage raw \
    --augmentation nozflip \
    --shuffled \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/nozflip_raw_shuffled \
    > ${DATA_ROOT}/nozflip_raw_shuffled/preparation.log


# for testing:
python scripts/preparation/prepare_dataset.py \
    --storage memmap \
    --augmentation none \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/none_memmap \
    > ${DATA_ROOT}/none_memmap/preparation.log
    # == non-augmented

python scripts/preparation/prepare_dataset.py \
    --storage raw \
    --augmentation none \
    ${DATA_ROOT}/original \
    ${DATA_ROOT}/none_raw \
    > ${DATA_ROOT}/none_raw/preparation.log
