import numpy as np
import time
import os
import re
import kaldiio
import torchaudio
import torch
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech
from espnet2.gan_tts.vits.loss import XvectorLoss

def get_xvector_dict(xv_dir):
    xv_dict = {}
    xv_name = os.listdir(xv_dir)
    for name in xv_name:
        xv_path = os.path.join(xv_dir, name)
        speaker = name.split('-')[0]
        if speaker not in xv_dict:
            xv_dict[speaker] = []
        xv_dict[speaker].append(xv_path)
    for speaker in xv_dict:
        xv_dict[speaker].sort()
    return xv_dict


def cosine_similarity_score(exp_wav_dir, xv_dict, criterion):
    score_text = []
    ground_truth_dir = "synthesis_sentences/gather_bilingual_compare_exp/ground_truth"
    ground_truth_list = os.listdir(ground_truth_dir)
    for speaker in xv_dict:
        score = f"{speaker}\n"
        xv_target = torch.from_numpy(np.load(xv_dict[speaker][0]))
        for speaker_name in ground_truth_list:
            if speaker_name.split('-')[0] == speaker:
                break
        waveform_target, fs = torchaudio.load(os.path.join(ground_truth_dir, speaker_name))
        wav_dir = os.path.join(exp_wav_dir, speaker)
        for wav_name in os.listdir(wav_dir):
            waveform, fs = torchaudio.load(os.path.join(wav_dir, wav_name))
            # cosine_similarity = criterion(waveform, xv_target, get_cosine_similarity=True)
            cosine_similarity = criterion.wav_wav_loss(waveform, waveform_target, get_cosine_similarity=True)
            score += f"{wav_name}\t{cosine_similarity}\n"
        score_text.append(score)
    with open(os.path.join(exp_wav_dir, "ecapa_cosine_similarity_score.txt"), 'w', encoding='utf-8')as f:
        f.write('\n'.join(score_text))


if __name__ == "__main__":
    save_dir = "synthesis_sentences/mb_jets/xmu-boy"
    os.makedirs(save_dir, exist_ok=True)
    tts = Text2Speech.from_pretrained(model_file="exp/xmu-boy/tts_multiband_jets/latest.pth", device='cpu')
    # criterion = XvectorLoss("/work/ysj/espnet/tools/extract_xvector_model/ecapa_16k_fbank_81_shard_vad50_bt128_valid2048_adam/10.pt")

    # lid=1
    # lid = np.asarray(lid)
    # xv = np.load(f"/work/ysj/espnet/egs2/ysj/tts1/exp/gather_bilingual/xvector/ShiJiang-00151-xvector.npy")
    # xv = kaldiio.load_mat("dump/cyw_chinese/xvector/tr_no_dev/xvector.3.ark:1070")

    sentence = [
        # "尊敬的投资者#3，网金B#2、医疗B#2、南方原油#2、轶金融B#2、电子B#2、环保B端#2、证券B#2、高贝塔B#2、高铁B级#2、智能B#2、地产B端#2、高铁B端#2、信息安B#2、新能B#2、信息B#2、券商B#2、原油基金#2、嘉实原油#2、国泰商品交易价格#2高于基金份额净值#4。",
        # "二是#2 IPO #1提速#1打击#1炒壳#3纳入#1 MSCI 等#2制度#1环境#1变革#3引发的#1估值#1体系#1重构#4。",
        # "无论是#1银行#2自身#1结售汇#1差额#3还是#1银行#2代客#1结售汇#1差额#3都#1呈现出#2逆差#1不断#1扩大的#1趋势#4，",
        # "fuck #3 what do you wanna to do ?",
        # "二零#1零九#1年时#3号称#1要#1装上#1T#1—九十的#2 LEDS #1—一百#1五十#2主动#1防卫#1系统#3也#1未见#1踪影#4；",
        # "九六#1 B #1使用的#3则是#1先进#1一代的#2液力#1机械#1传动#1系统#3和#1方向盘#1驾驶#4",
        "基金ABC。",
        "can you 讲中文吗？",
        # "飘飘然#2就把#1小时候#2母亲#1一边#1哄他#1睡觉#3一边#1甜蜜#1回忆的#2那段#1秘辛#1说了#1出来#4，",
        # "而#1之前#2疯狂#1攻击的#1变异#1槐树#3却像是#1遇到#1天敌#1一般#2收回#1所有#1树根#2瑟瑟#1发抖#4。",
        # "您有一条开线记录#2 cassette #2已上#1 port #3，机台#3 T #1 H #1 C #1 V #1 D #2幺#1零#1零#3，产品#3 T #1 H #2七#1四#1五#2 A #1三#1 A #1 B #2幺#1零#1零#3，cassette #2 I #1 D维#3 T #1 J #1 A #1 G #2幺#1幺#4。",
        # "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？我是马可·波罗的远房亲戚吗？当然不是。在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。这大概就是我要讲的故事。我努力搜索自己的记忆，找寻自己对中国感兴趣的始点。翻开我的图书收藏，我找到了第一本有关中国的书。这本书是在1979年得到的，当时我24岁，巴西一家出版社出版了葡萄牙文版的《毛泽东选集》，这是我接触到的第一批中文书籍之一。此外，毛泽东的一些文章和演讲稿我也反复阅读了很多次，基本都能熟记于心。直到今天，我还时不时地拿出来翻阅，并总能在这位令人钦佩的中国领导人的著作中有新的收获。在我看来，中国革命及长征是一部真正的史诗。在历史的伟大时刻，正是由于他们的勇气和坚持，人类历史的进程才得以改变。",
        "您有一条开线记录~cassette~已上~port，机台~T~H~C~V~D~幺~零~零~，产品~T~H~七~四~五~A~三#~A~B~幺~零~零，cassette~ID~维~T~J~A~G~幺~幺。",
        "你好，我来自厦门天聪智能软件有限公司。",
        "又是连续两周的雷暴雨，真是好烦啊！",
        "很多人都在问我，你对中国的热情是如何产生的？毕竟，上世纪初移民到巴西的意大利人的后代与中国有什么关系呢？",
        "在童年和青春期，我从未接触过中国文化，但当我成年以后，却爱上了一个如此遥远、与我们的文化如此不同的国家。",
        "I see nobody but you",
        "软件园二期望海路二十一号之一三零一。 ",
        "有一天，森林里的小动物——小兔子捡到了一本好看的童话书。于是小兔子便把它捡到的童话故事书，交给了这里的村长。村长便把森林里的小动物都召集起来，看起了里面的故事。大家都夸小兔子是好样的。",
        "一天，小动物们要搞聚会了。小猴子拿出一颗豆子，小猴子把豆子放在桌子上，在阳光的照射下，小豆子活蹦乱跳。大家都鼓掌了。大家问小猴子豆子怎么会跳？小猴子掰开豆子里面有两条虫子，大家突然明白了。",
        "小猴子拿出一颗豆子。小猴子把豆子放在桌子上。",
        # "sp z ue5 g u n6 lp bb a ng6 l o k8 sp d ng1 d io ng1 np l ao2 t ua n2 g ui5 z oo3 sp h o np ng6 h o ng2 h ue1 sp l ia p7 iaN3 z o k7 p i n3 sp",
        # "sp i1 h oo6 sp h ua t7 h ia n6 np d i6 h e h7 sp g u6 s ioo ng6 sp e2 g i1 c oo3 d i ng1 bb i n6 np c u t7 h ia n6 s i n1 e2 np a p7 s ioo k7 s i ng5 sp e2 g u t7 z i t7 sp",
        # "同时③，实时①音视频①技术①在①远程①会诊②、远程①协诊②、远程①影像①方面的①应用③，不仅①可以①促进①更加①高效的①医医①协作①模式③，还①可以①推动①医联体①内①医疗①资源的①分级①协同①与①广泛①下沉④。",
        # "还①可以①推动①医联体内②医疗①资源的①分级①协同②与①广泛①下沉④。",
        # "爱意#1随风起#3，风止#1意难平#4。",
        # "截至2022年上半年，我国共累计设立2,050支政府引导基金，目标规模约12.82万亿元人民币，已认缴规模约6.39万亿元人民币。二十年发展以来，我国政府引导基金发生了哪些演变？存量优化新阶段下各地有何举措？主要省市情况如何？",
        # "重庆市9月16日通报确诊1例境外输入猴痘病例。这是此轮猴痘疫情流行以来，我国内地首次发现猴痘病例。（相关阅读：重庆市发现1例境外输入猴痘病例)。今年7月，世界卫生组织宣布猴痘成为国际关注的突发公共卫生事件。截至8月中旬，全球89个国家和地区报告了大约4万例病例。",
        # "《爱拼会赢》讲述：1979年春，在晋江，面朝一片贫瘠的土地，高进阳和叶守礼两个将近不惑之年的好兄弟，正在就如何致富、如何摆脱贫穷的问题争论得面红耳赤，彼时的高家和叶家都是拖儿带女、生计艰难。高进阳夫妇用一条小渔船，开始了他们商业经贸的创业生涯。高家大儿子高海生、儿媳叶大莲砸锅卖铁办起了服装厂，一步步从手工作坊发展为拥有几个分厂的服装企业。而固守传统、热爱土地的叶守礼始终根植农业种植生产的领域，通过传统与科技相结合，也走出了一条特色农业致富之路，他的女儿二莲则在晋江特色美食行业中成为了佼佼者。随着企业的做大做强，他们在致富路上经受了一次又一次的考验。高家和叶家各自的儿女们，跟随父辈的脚步，在各自擅长的行业里发展壮大，爱拼敢赢、诚信团结，开创出“晋江制造”的新天地 ",
    #     "~厦门的美食真的太多啦。单是沙茶面就可以反复吃，不同店的沙茶酱~浓度不同、配料不同，做出的沙茶面的口感也不同，但都有一个共同点：好吃！",
    #     "~厦门风光绮丽，名胜古迹众多。从鼓浪屿~到南普陀寺~再到充满南洋风情的骑楼老街，秀美的自然与人文风光，让这座山海相连的城市~被誉为海上花园、东方威尼斯，每年吸引着不计其数的年轻人前来旅游、消费、生活。涌动的青年力量~令厦门城市活力倍增，城市文化魅力不断增加，商业发展迅速。",
    #     "~多年来，厦门中华城~持续延展商业空间的多元属性，不仅携手青年共创潮流，更深度洞察厦门人的生活方式，调改丰富业态为更多人群需求~提供休闲商业生活提案，让中华城成为厦门的地标~和一扇文化展示窗口，呈现出传统与现代交织，新旧渐次融合，传承与分享厦门的城市趣味灵魂。",
    #     "~传统建筑与时尚潮流的碰撞，让这处百年骑楼老街~形成独具特色的氛围感，走在厦门中华城场内，一场场潮流品牌时尚大秀魅力上演，吸引市民游客观赏，场外的落日晚风城市派对、摇摆邂逅舞会、夏日奇妙手账市集，则聚集了大量潮流青年互动交流。",
    ]

    # xv_dict = get_xvector_dict("exp/gather_bilingual/xvector")
    # for speaker in xv_dict:
    #     os.makedirs(os.path.join(save_dir, speaker), exist_ok=True)
    #     xv = np.load(xv_dict[speaker][0])
    #     for text in sentence: 
    #         start = time.time()
    #         wav = tts(text=text, spembs=xv)["wav"]
    #         end = time.time()
    #         print(f"{text}\n{end - start} s")
    #         text = text.replace("#1", "").replace("#2", "").replace("#3", "").replace("#4", "")  
    #         sf.write(os.path.join(save_dir, speaker, f"{text[:10]}.wav"), wav.cpu().numpy(), tts.fs, "PCM_16")    
    # cosine_similarity_score(save_dir, xv_dict, criterion)

    # sentence = []
    # with open("wenben.txt", 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         if line != "":
    #             sentence.append(line.strip())

    for text in sentence:
        start = time.time()
        wav = tts(text=text)["wav"]
        end = time.time()
        print(f"{text}\n{end - start} s")
        text = text.replace("#1", "").replace("#2", "").replace("#3", "").replace("#4", "")  
        sf.write(os.path.join(save_dir, f"{text[:10]}.wav"), wav.cpu().numpy(), tts.fs, "PCM_16")



