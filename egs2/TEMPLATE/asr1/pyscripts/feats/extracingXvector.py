# -*- coding:utf-8 -*-

import sys, os
import logging
import argparse
import traceback
import time
import yaml,copy
import math
import numpy as np

import torch
sys.path.insert(0, '/work/ysj/espnet/tools/subtools/pytorch')

import libs.egs.egs as egs
import libs.training.optim as optim
import libs.training.lr_scheduler as learn_rate_scheduler
import libs.training.trainer as trainer
import libs.support.kaldi_common as kaldi_common
import libs.support.utils as utils
from  libs.support.logging_stdout import patch_logging_stream

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
        description="""Train xvector framework with pytorch.""",
        formatter_class=argparse.RawTextHelpFormatter,
        conflict_handler='resolve')

parser.add_argument("--stage", type=int, default=1,
                    help="The stage to control the start of training epoch (default 4).\n"
                         "    stage 0: Generate raw wav kaldidir which contains utt2chunk utt2sr and utt2dur. (preprocess_raw_wav_egs.sh).\n"
                         "    stage 1: remove utts (preprocess_raw_wav_egs.sh).\n"
                         "    stage 2.1: get chunk egs (preprocess_raw_wav_egs.sh).\n"
                         "    stage 2.2: Prepare speech augment csv files.\n"                          
                         "    stage 3: Prepare speech augment csv files.\n"                         
                         "    stage 4: extract xvector.")

parser.add_argument("--endstage", type=int, default=1,
                    help="The endstage to control the endstart of training epoch (default 5).")
parser.add_argument("--feats_dir", type=str, required=True, help="feats dir")
parser.add_argument("--xvectors_dir", type=str, required=True, help="xvectors dir")
parser.add_argument("--nj", type=int, default=32, help="num job")
args = parser.parse_args()

##--------------------------------------------------##
## Control options

stage = max(1, args.stage)
endstage = min(1, args.endstage)
##--------------------------------------------------##


suffix = "params" # Used in saved model file.
model_blueprint = "subtools/pytorch/model/resnet-se-xvector.py"
model_dir = "/work/ysj/espnet/tools/extract_xvector_model/res34se_fbank_81_shard_16k_random"
##--------------------------------------------------##

#### Extract xvector
if stage <= 1 <= endstage and utils.is_main_training():
    to_extracted_positions = ["near"] # Define this w.r.t extracted_embedding param of model_blueprint.
    to_extracted_epochs = ["8"] # It is model's name, such as 10.params or final.params (suffix is w.r.t package).

    nj = args.nj
    force = False
    use_gpu = True
    gpu_id = ""
    sleep_time = 10
    cmn=True
    # Run a batch extracting process.
    try:
        for position in to_extracted_positions:
            # Generate the extracting config from nnet config where 
            # which position to extract depends on the 'extracted_embedding' parameter of model_creation (by my design).
            model_blueprint, model_creation = utils.read_nnet_config("{0}/config/nnet.config".format(model_dir))
            model_blueprint = model_dir + '/config/resnet_se_xvector.py'
            model_creation = model_creation.replace("training=True", "training=False") # To save memory without loading some independent components.
            model_creation = model_creation.replace("extracted_embedding='far'", "extracted_embedding='near'")
            extract_config = "{0}.extract.config".format(position)
            utils.write_nnet_config(model_blueprint, model_creation, "{0}/config/{1}".format(model_dir, extract_config))
            for epoch in to_extracted_epochs:
                model_file = "{0}.{1}".format(epoch, suffix)
                point_name = "{0}_epoch_{1}".format(position, epoch)

                # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=4 
                # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
                model_path = "{0}/{1}".format(model_dir, model_file)
                while True:
                    if os.path.exists(model_path):
                        break
                    else:
                        time.sleep(sleep_time)

                datadir = args.feats_dir
                outdir = args.xvectors_dir
                # Use a well-optimized shell script (with multi-processes) to extract xvectors.
                # Another way: use subtools/splitDataByLength.sh and subtools/pytorch/pipeline/onestep/extract_embeddings.py 
                # with python's threads to extract xvectors directly, but the shell script is more convenient.
                kaldi_common.execute_command("bash subtools/pytorch/pipeline/extract_xvectors_for_pytorch.sh "
                                            "--model {model_file} --cmn {cmn} --nj {nj} --use-gpu {use_gpu} --gpu-id '{gpu_id}' "
                                            " --force {force} --nnet-config config/{extract_config} "
                                            "{model_dir} {datadir} {outdir}".format(model_file=model_file, cmn=str(cmn).lower(), nj=nj,
                                            use_gpu=str(use_gpu).lower(), gpu_id=gpu_id, force=str(force).lower(), extract_config=extract_config,
                                            model_dir=model_dir, datadir=datadir, outdir=outdir))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)