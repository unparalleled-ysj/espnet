import os
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech


save_dir = "synthesis_sentences"
os.makedirs(save_dir, exist_ok=True)
tts = Text2Speech.from_pretrained(model_file="exp/CustomerService/tts_vits/latest.pth")
wav = tts("今天，天气真好呀！")["wav"]
sf.write(os.path.join(save_dir, f"test.wav"), wav.numpy(), tts.fs, "PCM_16")


