import os
import click


def augmentDataByNoisePostProcess(map_dir, noise_fs):
    wav_scp = os.path.join(map_dir, 'wav.scp')
    wav_scp_for_xv = []
    wav_scp_for_train = []
    wav_scp_dict = {}
    with open(wav_scp, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.strip().split(' ')
            utt_id = content[0]
            cmd = content[1:]
            wav_scp_dict[utt_id]=' '.join(cmd)
            if utt_id.split('-')[-1] in ["reverb", "noise", "music", "babble"]:
                cmd[-3] = f"\"sox {cmd[-3]} -r {noise_fs} -t wav - |\""
            wav_scp_for_xv.append(utt_id + ' ' + ' '.join(cmd))
    
    for utt_id in wav_scp_dict:
        cmd = wav_scp_dict[utt_id]
        if utt_id.split('-')[-1] in ["reverb", "noise", "music", "babble"]:
            cmd = wav_scp_dict['-'.join(utt_id.split('-')[:-1])]
        wav_scp_for_train.append(utt_id + ' ' + cmd)

    with open(os.path.join(map_dir, 'wav.scp'), 'w', encoding='utf-8')as f:
        wav_scp_for_xv.sort()
        f.write('\n'.join(wav_scp_for_xv)+'\n')

    with open(os.path.join(map_dir, 'wav.scp.train'), 'w', encoding='utf-8')as f:
        wav_scp_for_train.sort()
        f.write('\n'.join(wav_scp_for_train)+'\n')


@click.command()
@click.option("--map_dir", type=str, help="kaldi map files dir")
@click.option("--noise_fs", type=int, default=16000, help="noise sample rate")
def main(
    map_dir: str,
    noise_fs: int=16000,
):
    augmentDataByNoisePostProcess(map_dir, noise_fs)


if __name__ == "__main__":
    main()