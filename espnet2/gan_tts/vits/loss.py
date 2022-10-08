# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS-related loss modules.

This code is based on https://github.com/jaywalnut310/vits.

"""

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi


class KLDivergenceLoss(torch.nn.Module):
    """KL divergence loss."""

    def forward(
        self,
        z_p: torch.Tensor,
        logs_q: torch.Tensor,
        m_p: torch.Tensor,
        logs_p: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate KL divergence loss.

        Args:
            z_p (Tensor): Flow hidden representation (B, H, T_feats).
            logs_q (Tensor): Posterior encoder projected scale (B, H, T_feats).
            m_p (Tensor): Expanded text encoder projected mean (B, H, T_feats).
            logs_p (Tensor): Expanded text encoder projected scale (B, H, T_feats).
            z_mask (Tensor): Mask tensor (B, 1, T_feats).

        Returns:
            Tensor: KL divergence loss.

        """
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        kl = torch.sum(kl * z_mask)
        loss = kl / torch.sum(z_mask)

        return loss


class XvectorLoss(torch.nn.Module):
    """xvector loss"""

    def __init__(
        self,
        model_path="/work/ysj/espnet/tools/extract_xvector_model/res34se_fbank_81_shard_16k_random/8.pt",
        feature_type="fbank",
        kaldi_featset={
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25,
            "low_freq": 40,
            "high_freq": -200,
            "energy_floor": 0.0,
            "dither": 0,
            "use_energy": True,
        },
        mean_var_conf={
            "mean_norm": True,
            "std_norm": False,
        },
    ):
        super().__init__()
        self.xv_model = torch.jit.load(model_path)
        self.xv_model.eval()
        self.feat_type = feature_type
        self.mean_var_conf = mean_var_conf
        self.kaldi_featset = kaldi_featset
    
    def forward(
        self,
        waveform,
        xv_target,
        get_cosine_similarity=False,
    ):
        self.xv_model.eval()
        feats = self.kaldi_features_extract(waveform)
        xv = self.xv_model.extract_embedding_whole(feats, "near")
        consine_score = F.cosine_similarity(xv, xv_target, dim=0)
        if get_cosine_similarity:
            return consine_score
        xv_loss = 1 - consine_score
        return xv_loss

    def wav_wav_loss(self, waveform, waveform_target, get_cosine_similarity=False):
        self.xv_model.eval()
        feats = self.kaldi_features_extract(waveform)
        xv = self.xv_model.extract_embedding_whole(feats, "near")
        feats_target = self.kaldi_features_extract(waveform_target)
        xv_target = self.xv_model.extract_embedding_whole(feats_target, "near")
        consine_score = F.cosine_similarity(xv, xv_target, dim=0)
        if get_cosine_similarity:
            return consine_score
        xv_loss = 1 - consine_score
        return xv_loss

    def kaldi_features_extract(self, wavforms):
        if len(wavforms.shape) == 1:
            wavforms=wavforms.unsqueeze(0)
        if self.feat_type == 'mfcc':
            feat = kaldi.mfcc(wavforms, **self.kaldi_featset)
        else:
            feat = kaldi.fbank(wavforms, **self.kaldi_featset)
        feat = self.InputSequenceNormalization(feat)
        return feat
    
    def InputSequenceNormalization(self, feat):
        if self.mean_var_conf['mean_norm']:
            mean = torch.mean(feat, dim=0).detach().data
        else:
            mean = torch.tensor([0.0], device=feat.device)

        # Compute current std
        if self.mean_var_conf['std_norm']:
            std = torch.std(feat, dim=0).detach().data
        else:
            std = torch.tensor([1.0], device=feat.device)

        # Improving numerical stability of std
        std = torch.max(
            std, 1e-10 * torch.ones_like(std)
        )
        feat = (feat - mean.data) / std.data

        return feat

if __name__ == "__main__":
    import kaldiio
    xv_target = torch.from_numpy(kaldiio.load_mat("dump/cyw_chinese/xvector/tr_no_dev/xvector.3.ark:1070"))
    xv_loss = XvectorLoss()
    waveform, fs = torchaudio.load("/work/ysj/espnet/egs2/ysj/tts1/synthesis_sentences/exhibit/ground_truth/cyw-chinese.wav")
    loss = xv_loss(waveform, xv_target)
    print(loss)
    # loss1 = xv_loss(waveform[:, :256*32], xv_target)
    # loss2 = xv_loss(waveform[:, :256*64], xv_target)
    # loss3 = xv_loss(waveform[:, :256*128], xv_target)
    # print(loss1, loss2, loss3)
