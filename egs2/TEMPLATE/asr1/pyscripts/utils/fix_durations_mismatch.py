import os
import click
import numpy as np

def fix_durations_mismatch(durations_file, feat_length_file):
    sum_durations_dict = {}
    durations_dict = {}
    with open(durations_file, 'r', encoding='utf-8')as f:
        for line in f:
            content = line.strip().split(' ')
            utt_id = content[0]
            duration = content[1:]
            durations_dict[utt_id] = duration
            sum_duration = 0
            for d in content[1:]:
                sum_duration += int(d)
            sum_durations_dict[utt_id] = sum_duration

    feat_length_list = {}
    with open(feat_length_file, 'r', encoding='utf-8')as f:
        for line in f:
            content = line.strip().split(' ')
            utt_id = content[0]
            feat_length = int(np.load(content[1]).tolist()[0])
            feat_length_list[utt_id] = feat_length
    
    for utt_id in feat_length_list:
        if sum_durations_dict[utt_id] != feat_length_list[utt_id]:
            diff = sum_durations_dict[utt_id] - feat_length_list[utt_id]
            durations_dict[utt_id][-3] = str(int(durations_dict[utt_id][-3]) - diff)
            assert int(durations_dict[utt_id][-3]) >= 0
    
    with open(durations_file, 'w', encoding='utf-8')as f:
        for utt_id in durations_dict:
            f.write(utt_id + ' ' + ' '.join(durations_dict[utt_id]) + '\n')
       

@click.command()
@click.option("--dataset", type=str, help="dataset name")
@click.option("--tts_stat_dir", type=str, help="tts stat dir name")
@click.option("--train_set", type=str, help="train set name")
@click.option("--valid_set", type=str, help="valid set name")
def main(
    dataset: str,
    tts_stat_dir: str,
    train_set: str,
    valid_set: str,
):
    for durations_set, feat_length_set in zip([train_set, valid_set], ['train', 'valid']):
        fix_durations_mismatch(f"data/{dataset}/{durations_set}/durations", f"{tts_stat_dir}/{feat_length_set}/collect_feats/feats_lengths.scp")


if __name__ == "__main__":
    main()

