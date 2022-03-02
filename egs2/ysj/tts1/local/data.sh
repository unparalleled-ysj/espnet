#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

dataset=$1
stage=0
stop_stage=1

train_set=tr_no_dev
train_dev=dev
recog_set=eval1

[ ! -e data/$dataset/train ] && mkdir -p data/$dataset/train

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    cp /work/ysj/TTS_TrainData/$dataset/kaldi_map_folder/spk2utt data/$dataset/train/spk2utt
    cp /work/ysj/TTS_TrainData/$dataset/kaldi_map_folder/utt2spk data/$dataset/train/utt2spk
    sed "s/wavs_origin/wavs/g" /work/ysj/TTS_TrainData/$dataset/kaldi_map_folder/wav.scp > data/$dataset/train/wav.scp
    cat /work/ysj/TTS_TrainData/$dataset/train.txt | awk -F "|" '{print $1"\t"$2}' > data/$dataset/train/text
    python pyscripts/utils/get_durations.py --durations_dir /work/ysj/TTS_TrainData/$dataset/durations --save_file data/$dataset/train/durations
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --last data/$dataset/train 100 data/$dataset/deveval
    utils/subset_data_dir.sh --last data/$dataset/deveval 50 data/$dataset/${recog_set}
    utils/subset_data_dir.sh --first data/$dataset/deveval 50 data/$dataset/${train_dev}
    n=$(( $(wc -l < data/$dataset/train/wav.scp) - 100 ))
    utils/subset_data_dir.sh --first data/$dataset/train ${n} data/$dataset/${train_set}

    cat data/$dataset/train/durations | tail -100 > data/$dataset/deveval/durations
    cat data/$dataset/deveval/durations | tail -50 > data/$dataset/${recog_set}/durations
    cat data/$dataset/deveval/durations | head -50 > data/$dataset/${train_dev}/durations
    cat data/$dataset/train/durations | head -${n} >  data/$dataset/${train_set}/durations
fi

log "Successfully finished. [elapsed=${SECONDS}s]"