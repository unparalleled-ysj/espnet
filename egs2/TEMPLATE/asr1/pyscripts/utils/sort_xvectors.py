import os
import click


def sort_xvectors(feats_dir, xvectors_dir):
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
            xvectors_dict[utt_id] = xvector_path
    
    sort_xvectors_list = []
    for utt_id in utt_id_list:
        sort_xvectors_list.append(utt_id + ' ' + xvectors_dict[utt_id])

    with open(os.path.join(xvectors_dir, "xvector.scp"), 'w', encoding='utf-8')as f:
        f.write('\n'.join(sort_xvectors_list)+'\n')




@click.command()
@click.option("--feats_dir", type=str, help="feats dir")
@click.option("--xvectors_dir", type=str, help="xvectors dir")
def main(
    feats_dir: str,
    xvectors_dir: str,
):
    sort_xvectors(feats_dir, xvectors_dir)

if __name__ == "__main__":
    main()