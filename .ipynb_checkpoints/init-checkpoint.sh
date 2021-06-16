#!/usr/bin/env bash
ROOT=/home/tione/notebook
DATASET_ROOT=${ROOT}/dataset
cd ${ROOT}
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "DATASET_ROOT= ${DATASET_ROOT}, not exists, mkdir"
    mkdir dataset
    cd dataset && mkdir frames
    cd ${ROOT}
fi
mkdir pretrained
mkdir checkpoint

cp -r ${ROOT}/algo-2021/dataset/label_id.txt ${ROOT}/algo-2021/dataset/tagging/ ${DATASET_ROOT}
cp -r ${ROOT}/TAAC-2021/scripts/txt_files/* ${DATASET_ROOT}/tagging/GroundTruth/datafile/
pip install coscmd
pip install coscmd -U
coscmd config -a AKIDUmEI59lYuzGH4e7078G9ljQ8anlSEACz -s zjDSovjQt40fiItDF2Y0BAsJfkQrn8C1 -b taac1-1300501903 -r ap-guangzhou

coscmd download -r pretrained/ ${ROOT}/pretrained/
coscmd download -r features/train_5k ${DATASET_ROOT}/frames/train_5k
coscmd download -r features/test_5k ${DATASET_ROOT}/frames/test_5k
coscmd download checkpoint/50_wp.pt ${ROOT}/checkpoint/