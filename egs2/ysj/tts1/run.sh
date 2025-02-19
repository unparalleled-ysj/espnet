#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# fs=44100
# n_fft=2048
# n_shift=512
# win_length=null

fs=16000
n_fft=1024
n_shift=256
win_length=null

opts=
if [ "${fs}" -eq 48000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

dataset="TalentedSoft_multilingual"
train_set=tr_no_dev
valid_set=dev
test_sets="dev eval1"

# train_config=conf/tuning/finetune_joint_conformer_fastspeech2_hifigan.yaml
# train_config=conf/tuning/train_conformer_fastspeech2.yaml
# train_config=conf/tuning/train_xvector_jets.yaml
# train_config=conf/tuning/finetune_multi_spk_vits.yaml
train_config=conf/tuning/train_multistream_istft_jets.yaml
# train_config=conf/tuning/train_simple_jets.yaml
inference_config=conf/tuning/decode_vits.yaml


g2p=none
vits_g2p=add_blank

# # train vits
# ./tts.sh \
#     --tts_task gan_tts \
#     --ngpu 1 \
#     --dataset $dataset \
#     --dumpdir dump/$dataset \
#     --expdir exp/$dataset \
#     --tts_stats_dir exp/$dataset/tts_stats_vits \
#     --tts_exp exp/$dataset/tts_multiband_xv_vits \
#     --use_xvector true \
#     --use_sid false \
#     --use_lid false \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --feats_extract linear_spectrogram \
#     --feats_normalize none \
#     --cleaner none \
#     --g2p "${vits_g2p}" \
#     --train_config "${train_config}" \
#     --inference_config "${inference_config}" \
#     --inference_model latest.pth \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/$dataset/${train_set}/text" \
#     ${opts} "$@"

# train fastspeech2

# ./tts.sh \
#     --tts_task gan_tts \
#     --ngpu 1 \
#     --dataset $dataset \
#     --dumpdir dump/$dataset \
#     --expdir exp/$dataset \
#     --tts_stats_dir exp/$dataset/tts_stat_fs2 \
#     --tts_exp exp/$dataset/tts_ft_joint_cfs2_hfg \
#     --lang zh \
#     --feats_type raw \
#     --fs "${fs}" \
#     --n_fft "${n_fft}" \
#     --n_shift "${n_shift}" \
#     --win_length "${win_length}" \
#     --cleaner none \
#     --g2p "${g2p}" \
#     --train_config "${train_config}" \
#     --inference_model train.text2mel_loss.best.pth \
#     --teacher_dumpdir data/$dataset \
#     --write_collected_feats true \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --srctexts "data/$dataset/${train_set}/text" \
#     ${opts} "$@"


# train jets
./tts.sh \
    --tts_task gan_tts \
    --ngpu 1 \
    --dataset $dataset \
    --dumpdir dump/$dataset \
    --expdir exp/$dataset \
    --tts_stats_dir exp/$dataset/tts_stats_jets \
    --tts_exp exp/$dataset/tts_ms_istft_jets_new \
    --use_xvector false \
    --use_sid true \
    --use_lid false \
    --fmax null \
    --fmin 0 \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_model latest.pth \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/$dataset/${train_set}/text" \
    ${opts} "$@"