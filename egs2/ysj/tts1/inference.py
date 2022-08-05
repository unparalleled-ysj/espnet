import numpy as np
import time
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
    save_dir = "synthesis_sentences/mb_jets/multiband_vits"
    os.makedirs(save_dir, exist_ok=True)
    tts = Text2Speech.from_pretrained(model_file="exp/CS_prosody_16k/tts_multiband_vits_add_blank/latest.pth")

    # lid=1
    # lid = np.asarray(lid)
    # xv = np.load(f"/work/ysj/espnet/egs2/ysj/tts1/exp/gather_bilingual/xvector/ShiJiang-00151-xvector.npy")

    # text = "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？我是马可·波罗的远房亲戚吗？当然不是。在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。这大概就是我要讲的故事。我努力搜索自己的记忆，找寻自己对中国感兴趣的始点。翻开我的图书收藏，我找到了第一本有关中国的书。这本书是在1979年得到的，当时我24岁，巴西一家出版社出版了葡萄牙文版的《毛泽东选集》，这是我接触到的第一批中文书籍之一。此外，毛泽东的一些文章和演讲稿我也反复阅读了很多次，基本都能熟记于心。直到今天，我还时不时地拿出来翻阅，并总能在这位令人钦佩的中国领导人的著作中有新的收获。在我看来，中国革命及长征是一部真正的史诗。在历史的伟大时刻，正是由于他们的勇气和坚持，人类历史的进程才得以改变。"
    # text = "很多人#1都在#1问我#1你对#1中国的#1热情#2是#1如何#1产生的#2毕竟#1上世纪#1初#1移民到#1巴西的#1意大利人的#1后代#2与#1中国#1有#1什么#1关系呢#4"
    # with open("/work/ysj/espnet/egs2/ysj/tts1/synthesis_sentences/english.txt", 'r', encoding='utf-8')as f:
    #     # split_texts = text_segment(text, split_text=[])
    #     split_texts = f.readlines()
    #     final_audio = tts(text=split_texts[0])["wav"].numpy()
    #     for split_text in split_texts[1:]:
    #         wav = tts(text=split_text)["wav"]
    #         final_audio = np.concatenate((final_audio, wav.numpy()), axis=0)
    #     sf.write(os.path.join(save_dir, f"promo_copy.wav"), final_audio, tts.fs, "PCM_16")

    sentence = [
        # "尊敬的投资者#3，网金B#2、医疗B#2、南方原油#2、轶金融B#2、电子B#2、环保B端#2、证券B#2、高贝塔B#2、高铁B级#2、智能B#2、地产B端#2、高铁B端#2、信息安B#2、新能B#2、信息B#2、券商B#2、原油基金#2、嘉实原油#2、国泰商品交易价格#2高于基金份额净值#4。",
        "二是#2 IPO #1提速#1打击#1炒壳#3纳入#1 MSCI 等#2制度#1环境#1变革#3引发的#1估值#1体系#1重构#4。",
        "无论是#1银行#2自身#1结售汇#1差额#3还是#1银行#2代客#1结售汇#1差额#3都#1呈现出#2逆差#1不断#1扩大的#1趋势#4，",
        "fuck #3 what do you wanna to do ?",
        "二零#1零九#1年时#3号称#1要#1装上#1T#1—九十的#2 LEDS #1—一百#1五十#2主动#1防卫#1系统#3也#1未见#1踪影#4；",
        "九六#1 B #1使用的#3则是#1先进#1一代的#2液力#1机械#1传动#1系统#3和#1方向盘#1驾驶#4",
        # "基金ABC。",
        # "can you 讲中文吗？",
        "飘飘然#2就把#1小时候#2母亲#1一边#1哄他#1睡觉#3一边#1甜蜜#1回忆的#2那段#1秘辛#1说了#1出来#4，",
        "而#1之前#2疯狂#1攻击的#1变异#1槐树#3却像是#1遇到#1天敌#1一般#2收回#1所有#1树根#2瑟瑟#1发抖#4。",
        "您有一条开线记录#2 cassette #2已上#1 port #3，机台#3 T #1 H #1 C #1 V #1 D #2幺#1零#1零#3，产品#3 T #1 H #2七#1四#1五#2 A #1三#1 A #1 B #2幺#1零#1零#3，cassette #2 I #1 D维#3 T #1 J #1 A #1 G #2幺#1幺#4。"
        # "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？我是马可·波罗的远房亲戚吗？当然不是。在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。这大概就是我要讲的故事。我努力搜索自己的记忆，找寻自己对中国感兴趣的始点。翻开我的图书收藏，我找到了第一本有关中国的书。这本书是在1979年得到的，当时我24岁，巴西一家出版社出版了葡萄牙文版的《毛泽东选集》，这是我接触到的第一批中文书籍之一。此外，毛泽东的一些文章和演讲稿我也反复阅读了很多次，基本都能熟记于心。直到今天，我还时不时地拿出来翻阅，并总能在这位令人钦佩的中国领导人的著作中有新的收获。在我看来，中国革命及长征是一部真正的史诗。在历史的伟大时刻，正是由于他们的勇气和坚持，人类历史的进程才得以改变。",
        # "你好，我来自厦门天聪智能软件有限公司。",
        # "又是连续两周的雷暴雨，真是好烦啊！",
        # "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？",
        # "在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。",
        # "I see nobody but you",
        # "软件园二期望海路二十一号之一三零一。 ",
        # "有一天，森林里的小动物——小兔子捡到了一本好看的童话书。于是小兔子便把它捡到的童话故事书，交给了这里的村长。村长便把森林里的小动物都召集起来，看起了里面的故事。大家都夸小兔子是好样的。",
        # "一天，小动物们要搞聚会了。小猴子拿出一颗豆子，小猴子把豆子放在桌子上，在阳光的照射下，小豆子活蹦乱跳。大家都鼓掌了。大家问小猴子豆子怎么会跳？小猴子掰开豆子里面有两条虫子，大家突然明白了。",
        # "小猴子拿出一颗豆子。小猴子把豆子放在桌子上。",
    ]

    for text in sentence: 
        start = time.time()
        wav = tts(text=text)["wav"]
        end = time.time()
        print(f"{text}\n{end - start} s")
        text = text.replace("#1", "").replace("#2", "").replace("#3", "").replace("#4", "")  
        sf.write(os.path.join(save_dir, f"{text[:10]}.wav"), wav.numpy(), tts.fs, "PCM_16")



