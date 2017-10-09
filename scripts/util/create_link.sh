#!/usr/bin/env bash

create_link ()
{
    cd "${1}"
    ln -s ../tianchi_positive_samples/merged subset10
    cd ..
}

cd "../data"

create_link "v2_dice_memmap_shuffled"
create_link "v2_downsampledA"
create_link "v2_downsampledB"
create_link "v2_downsampledC"
create_link "v2_fonova100"
create_link "v2_fonova100_high_res"
create_link "v2_fonova25_high_res"
create_link "v2_fonova7"
create_link "v2_fonova7_high_res"
create_link "v2_kokA"
create_link "v2_kokB"
create_link "v2_nozflip_memmap_shuffled"
create_link "v2_xyA"
create_link "v2_xyB"
create_link "v2_xyC"
create_link "v2_xyD"
create_link "v2_xyE"
