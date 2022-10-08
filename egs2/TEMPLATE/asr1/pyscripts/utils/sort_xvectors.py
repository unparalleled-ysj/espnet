import os
import click
import random

def sort_xvectors(feats_dir, xvectors_dir, shuffle):
    utt_id_list = []
    with open(os.path.join(feats_dir, "utt2spk"), 'r', encoding='utf-8')as f:
        utt_id = f.readline().strip().split(' ')[0]
        while utt_id:
            utt_id_list.append(utt_id)
            utt_id = f.readline().strip().split(' ')[0]
    
    xvectors_dict = {}
    with open(os.path.join(xvectors_dir, "xvector.scp"), 'r', encoding='utf-8')as f:
        for line in f:
            content = line.strip().split(' ')
            utt_id = content[0]
            xvector_path = content[1]
            if shuffle:
                speaker = '-'.join(utt_id.split('-')[:-1]) if utt_id.split('-')[-1] not in ["reverb", "noise", "music", "babble"] else '-'.join(utt_id.split('-')[:-2])
                if speaker not in xvectors_dict:
                    xvectors_dict[speaker] = []
                xvectors_dict[speaker].append(xvector_path)
            else:
                xvectors_dict[utt_id] = xvector_path
    
    sort_xvectors_list = []
    for utt_id in utt_id_list:
        if shuffle:
            speaker = '-'.join(utt_id.split('-')[:-1]) if utt_id.split('-')[-1] not in ["reverb", "noise", "music", "babble"] else '-'.join(utt_id.split('-')[:-2])
            xvector_path = random.choice(xvectors_dict[speaker])
            sort_xvectors_list.append(utt_id + ' ' + xvector_path)
            xvectors_dict[speaker].remove(xvector_path)
        else:
            sort_xvectors_list.append(utt_id + ' ' + xvectors_dict[utt_id])

    with open(os.path.join(xvectors_dir, "xvector.scp"), 'w', encoding='utf-8')as f:
        f.write('\n'.join(sort_xvectors_list)+'\n')





@click.command()
@click.option("--feats_dir", type=str, help="feats dir")
@click.option("--xvectors_dir", type=str, help="xvectors dir")
@click.option("--shuffle", type=bool, default=False, help="whether shuffle xvector of same speaker")
def main(
    feats_dir: str,
    xvectors_dir: str,
    shuffle: bool=False,
):
    sort_xvectors(feats_dir, xvectors_dir, shuffle)

if __name__ == "__main__":
    main()