#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000
n_fft=400
n_shift=160
win_length=null

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

dataset="CustomerService"
train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

train_config=conf/tuning/train_joint_conformer_fastspeech2_hifigan.yaml
# train_config=conf/tuning/train_conformer_fastspeech2.yaml
# train_config=conf/tuning/train_vits.yaml
# inference_config=conf/tuning/decode_vits.yaml

g2p=none

# train vits
# ./tts.sh \
#     --tts_task gan_tts \
#     --ngpu 2 \
#     --dataset $dataset \
#     --dumpdir dump/$dataset \
#     --expdir exp/$dataset \
#     --tag vits \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --feats_extract linear_spectrogram \
#     --feats_normalize none \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --inference_model latest.pth \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/$dataset/${train_set}/text" \
#     ${opts} "$@"

# train fastspeech2

./tts.sh \
    --tts_task gan_tts \
    --ngpu 2 \
    --dataset $dataset \
    --dumpdir dump/$dataset \
    --expdir exp/$dataset \
    --tts_stats_dir exp/$dataset/tts_stat_fs2 \
    --tts_exp exp/$dataset/tts_joint_cfs2_hfg \
    --lang zh \
    --feats_type raw \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --teacher_dumpdir data/$dataset \
    --write_collected_feats true \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/$dataset/${train_set}/text" \
    ${opts} "$@"