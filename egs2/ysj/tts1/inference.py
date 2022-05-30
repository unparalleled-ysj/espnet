import numpy as np
import os
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech

split_character = ['，', '。', '？', '！', ' ', '、', '；', '：']
def text_segment(text, position=60, max_len=60, split_text=[]):
    if position >= len(text):
        if text[position - max_len:] != '':
            split_text.append(text[position - max_len:])
        return split_text
    else:
        for i in range(position, position - max_len, -1):
            if i == position - max_len + 1:
                split_text.append(text[position - max_len:position])
                return text_segment(text, position + max_len, max_len, split_text)
            if text[i] in split_character:
                split_position = i + 1
                split_text.append(text[position - max_len:split_position])
                return text_segment(text, split_position + max_len, max_len, split_text)


if __name__ == "__main__":
    save_dir = "synthesis_sentences"
    os.makedirs(save_dir, exist_ok=True)
    tts = Text2Speech.from_pretrained(model_file="exp/obm_prosody/tts_vits/latest.pth")
    # lid=1
    # lid = np.asarray(lid)
    # xv = np.load(f"/work/ysj/TTS_TrainData/TS_gather/xvector/ShiJiang-00151-xvector.npy")

    # text = "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？我是马可·波罗的远房亲戚吗？当然不是。在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。这大概就是我要讲的故事。我努力搜索自己的记忆，找寻自己对中国感兴趣的始点。翻开我的图书收藏，我找到了第一本有关中国的书。这本书是在1979年得到的，当时我24岁，巴西一家出版社出版了葡萄牙文版的《毛泽东选集》，这是我接触到的第一批中文书籍之一。此外，毛泽东的一些文章和演讲稿我也反复阅读了很多次，基本都能熟记于心。直到今天，我还时不时地拿出来翻阅，并总能在这位令人钦佩的中国领导人的著作中有新的收获。在我看来，中国革命及长征是一部真正的史诗。在历史的伟大时刻，正是由于他们的勇气和坚持，人类历史的进程才得以改变。"
    text = "武 汉 市 #2 长 江 大 桥 #4"

    # split_texts = text_segment(text, split_text=[])
    # final_audio = tts(text=split_texts[0])["wav"].numpy()
    # for split_text in split_texts[1:]:
    #     wav = tts(text=split_text)["wav"]
    #     final_audio =  np.concatenate((final_audio, wav.numpy()), axis=0)
    # sf.write(os.path.join(save_dir, f"obm_concat.wav"), final_audio, tts.fs, "PCM_16")

    wav = tts(text=text)["wav"]
    sf.write(os.path.join(save_dir, f"obm_prosody.wav"), wav.numpy(), tts.fs, "PCM_16")


