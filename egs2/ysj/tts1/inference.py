import numpy as np
import os
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech


save_dir = "synthesis_sentences/qc_id"
os.makedirs(save_dir, exist_ok=True)
tts = Text2Speech.from_pretrained(model_file="exp/qc/tts_id_vits/3epoch.pth")
# spk = "gu"
# for idx in range(1, 5):
    # xv = np.load(f"/work/ysj/TTS_TrainData/TS_record/{spk}/xvector/{spk}-0000{idx}-xvector.npy")
for sid in [250, 251, 252, 253]:
    id = np.asarray(sid)
    wav = tts(text="你好，我来自于上海企创信息科技有限公司，很高兴认识大家。", sids=id)["wav"]
    sf.write(os.path.join(save_dir, f"{id}.wav"), wav.numpy(), tts.fs, "PCM_16")


