import click
import numpy as np
import os


def get_durations_from_dir(durations_dir, save_file):
    durations_list = []
    for file_name in os.listdir(durations_dir):
        utt_id = '-'.join(file_name.split('-')[:-1])
        duration = ' '.join(list(map(str, np.load(os.path.join(durations_dir, file_name)).tolist())))
        durations_list.append(f"{utt_id} {duration}")
    with open(save_file, 'w', encoding='utf-8')as f:
        durations_list.sort()
        f.write('\n'.join(durations_list) + '\n')


@click.command()
@click.option("--durations_dir", type=str, help="durations dir path")
@click.option("--save_file", type=str, help="save file path")
def main(
    durations_dir: str,
    save_file: str
):
    get_durations_from_dir(durations_dir, save_file)


if __name__ == "__main__":
    main()