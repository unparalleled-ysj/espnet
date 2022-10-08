# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""VITS module for GAN-TTS task."""

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, Dict, Optional

import torch
from typeguard import check_argument_types

from espnet2.gan_tts.abs_gan_tts import AbsGANTTS
from espnet2.gan_tts.hifigan import (
    HiFiGANMultiPeriodDiscriminator,
    HiFiGANMultiScaleDiscriminator,
    HiFiGANMultiScaleMultiPeriodDiscriminator,
    HiFiGANPeriodDiscriminator,
    HiFiGANScaleDiscriminator,
)
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
    MelSpectrogramLoss,
)
from espnet2.gan_tts.bigvgan import BigVGANMultiPeriodDiscriminator
from espnet2.gan_tts.utils import get_segments
from espnet2.gan_tts.vits.generator import VITSGenerator
from espnet2.gan_tts.melgan.pqmf import PQMF
from espnet2.gan_tts.vits.loss import KLDivergenceLoss, XvectorLoss
from espnet2.gan_tts.melgan.stft_loss import MultiResolutionSTFTLoss
from espnet2.torch_utils.device_funcs import force_gatherable

AVAILABLE_GENERATERS = {
    "vits_generator": VITSGenerator,
}
AVAILABLE_DISCRIMINATORS = {
    "hifigan_period_discriminator": HiFiGANPeriodDiscriminator,
    "hifigan_scale_discriminator": HiFiGANScaleDiscriminator,
    "hifigan_multi_period_discriminator": HiFiGANMultiPeriodDiscriminator,
    "hifigan_multi_scale_discriminator": HiFiGANMultiScaleDiscriminator,
    "hifigan_multi_scale_multi_period_discriminator": HiFiGANMultiScaleMultiPeriodDiscriminator,  # NOQA
    "bigvgan_multi_period_discriminator": BigVGANMultiPeriodDiscriminator,
}

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class VITS(AbsGANTTS):
    """VITS module (generator + discriminator).

    This is a module of VITS described in `Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        # generator related
        idim: int,
        odim: int,
        sampling_rate: int = 22050,
        generator_type: str = "vits_generator",
        generator_params: Dict[str, Any] = {
            "hidden_channels": 192,
            "spks": None,
            "langs": None,
            "spk_embed_dim": None,
            "global_channels": -1,
            "segment_size": 32,
            "text_encoder_attention_heads": 2,
            "text_encoder_ffn_expand": 4,
            "text_encoder_blocks": 6,
            "text_encoder_positionwise_layer_type": "conv1d",
            "text_encoder_positionwise_conv_kernel_size": 1,
            "text_encoder_positional_encoding_layer_type": "rel_pos",
            "text_encoder_self_attention_layer_type": "rel_selfattn",
            "text_encoder_activation_type": "swish",
            "text_encoder_normalize_before": True,
            "text_encoder_dropout_rate": 0.1,
            "text_encoder_positional_dropout_rate": 0.0,
            "text_encoder_attention_dropout_rate": 0.0,
            "text_encoder_conformer_kernel_size": 7,
            "use_macaron_style_in_text_encoder": True,
            "use_conformer_conv_in_text_encoder": True,
            "decoder_type": "hifigan",
            "decoder_kernel_size": 7,
            "decoder_channels": 512,
            "decoder_out_channels": 1,
            "decoder_upsample_scales": [8, 8, 2, 2],
            "decoder_upsample_kernel_sizes": [16, 16, 4, 4],
            "decoder_resblock_kernel_sizes": [3, 7, 11],
            "decoder_resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "use_weight_norm_in_decoder": True,
            "posterior_encoder_kernel_size": 5,
            "posterior_encoder_layers": 16,
            "posterior_encoder_stacks": 1,
            "posterior_encoder_base_dilation": 1,
            "posterior_encoder_dropout_rate": 0.0,
            "use_weight_norm_in_posterior_encoder": True,
            "flow_flows": 4,
            "flow_kernel_size": 5,
            "flow_base_dilation": 1,
            "flow_layers": 4,
            "flow_dropout_rate": 0.0,
            "use_weight_norm_in_flow": True,
            "use_only_mean_in_flow": True,
            "stochastic_duration_predictor_kernel_size": 3,
            "stochastic_duration_predictor_dropout_rate": 0.5,
            "stochastic_duration_predictor_flows": 4,
            "stochastic_duration_predictor_dds_conv_layers": 3,
        },
        use_pqmf: bool = False,
        pqmf_params: Dict[str, Any] = {
            "subbands": 4,
            "taps": 62,
            "cutoff_ratio": 0.142,
            "beta": 9.0,
        },
        # discriminator related
        discriminator_type: str = "hifigan_multi_scale_multi_period_discriminator",
        discriminator_params: Dict[str, Any] = {
            "scales": 1,
            "scale_downsample_pooling": "AvgPool1d",
            "scale_downsample_pooling_params": {
                "kernel_size": 4,
                "stride": 2,
                "padding": 2,
            },
            "scale_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [15, 41, 5, 3],
                "channels": 128,
                "max_downsample_channels": 1024,
                "max_groups": 16,
                "bias": True,
                "downsample_scales": [2, 2, 4, 4, 1],
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
            "follow_official_norm": False,
            "periods": [2, 3, 5, 7, 11],
            "period_discriminator_params": {
                "in_channels": 1,
                "out_channels": 1,
                "kernel_sizes": [5, 3],
                "channels": 32,
                "downsample_scales": [3, 3, 3, 3, 1],
                "max_downsample_channels": 1024,
                "bias": True,
                "nonlinear_activation": "LeakyReLU",
                "nonlinear_activation_params": {"negative_slope": 0.1},
                "use_weight_norm": True,
                "use_spectral_norm": False,
            },
        },
        # loss related
        use_xv_loss: bool = False,
        xvector_loss_params: Dict[str, Any] = {
            "model_path": "/work/ysj/espnet/tools/extract_xvector_model/res34se_fbank_81_shard_16k_random/8.pt",
            "feature_type": "fbank",
            "kaldi_featset": {
                "num_mel_bins": 80,
                "frame_shift": 10,
                "frame_length": 25,
                "low_freq": 40,
                "high_freq": -200,
                "energy_floor": 0.0,
                "dither": 0,
                "use_energy": True,
            },
            "mean_var_conf": {
                "mean_norm": True,
                "std_norm": False,
            },
        },
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        mel_loss_params: Dict[str, Any] = {
            "fs": 22050,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": None,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        lambda_adv: float = 1.0,
        lambda_mel: float = 45.0,
        lambda_feat_match: float = 2.0,
        lambda_dur: float = 1.0,
        lambda_kl: float = 1.0,
        cache_generator_outputs: bool = True,
    ):
        """Initialize VITS module.

        Args:
            idim (int): Input vocabrary size.
            odim (int): Acoustic feature dimension. The actual output channels will
                be 1 since VITS is the end-to-end text-to-wave model but for the
                compatibility odim is used to indicate the acoustic feature dimension.
            sampling_rate (int): Sampling rate, not used for the training but it will
                be referred in saving waveform during the inference.
            generator_type (str): Generator type.
            generator_params (Dict[str, Any]): Parameter dict for generator.
            discriminator_type (str): Discriminator type.
            discriminator_params (Dict[str, Any]): Parameter dict for discriminator.
            generator_adv_loss_params (Dict[str, Any]): Parameter dict for generator
                adversarial loss.
            discriminator_adv_loss_params (Dict[str, Any]): Parameter dict for
                discriminator adversarial loss.
            feat_match_loss_params (Dict[str, Any]): Parameter dict for feat match loss.
            mel_loss_params (Dict[str, Any]): Parameter dict for mel loss.
            lambda_adv (float): Loss scaling coefficient for adversarial loss.
            lambda_mel (float): Loss scaling coefficient for mel spectrogram loss.
            lambda_feat_match (float): Loss scaling coefficient for feat match loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_kl (float): Loss scaling coefficient for KL divergence loss.
            cache_generator_outputs (bool): Whether to cache generator outputs.

        """
        assert check_argument_types()
        super().__init__()
        self.use_pqmf = use_pqmf
        self.use_xv_loss = use_xv_loss

        # define modules
        generator_class = AVAILABLE_GENERATERS[generator_type]
        if generator_type == "vits_generator":
            # NOTE(kan-bayashi): Update parameters for the compatibility.
            #   The idim and odim is automatically decided from input data,
            #   where idim represents #vocabularies and odim represents
            #   the input acoustic feature dimension.
            generator_params.update(vocabs=idim, aux_channels=odim)
        self.generator = generator_class(
            **generator_params,
        )
        if self.use_pqmf:
            self.pqmf = PQMF(**pqmf_params)
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[1024, 2048, 512],
                hop_sizes=[120, 240, 50],
                win_lengths=[600, 1200, 240],
                window="hann_window",
            )
            self.sub_stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=[384, 683, 171],
                hop_sizes=[30, 60, 10],
                win_lengths=[150, 300, 60],
                window="hann_window",
            )
        discriminator_class = AVAILABLE_DISCRIMINATORS[discriminator_type]
        if self.use_xv_loss:
            self.xv_loss = XvectorLoss(
                **xvector_loss_params,
            )
        self.discriminator = discriminator_class(
            **discriminator_params,
        )
        self.generator_adv_loss = GeneratorAdversarialLoss(
            **generator_adv_loss_params,
        )
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params,
        )
        self.feat_match_loss = FeatureMatchLoss(
            **feat_match_loss_params,
        )
        self.mel_loss = MelSpectrogramLoss(
            **mel_loss_params,
        )
        self.kl_loss = KLDivergenceLoss()

        # coefficients
        self.lambda_adv = lambda_adv
        self.lambda_mel = lambda_mel
        self.lambda_kl = lambda_kl
        self.lambda_feat_match = lambda_feat_match
        self.lambda_dur = lambda_dur

        # cache
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # store sampling rate for saving wav file
        # (not used for the training)
        self.fs = sampling_rate

        # store parameters for test compatibility
        self.spks = self.generator.spks
        self.langs = self.generator.langs
        self.spk_embed_dim = self.generator.spk_embed_dim

    @property
    def require_raw_speech(self):
        """Return whether or not speech is required."""
        return True

    @property
    def require_vocoder(self):
        """Return whether or not vocoder is required."""
        return False

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        forward_generator: bool = True,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            forward_generator (bool): Whether to forward generator.

        Returns:
            Dict[str, Any]:
                - loss (Tensor): Loss scalar tensor.
                - stats (Dict[str, float]): Statistics to be monitored.
                - weight (Tensor): Weight tensor to summarize losses.
                - optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        if forward_generator:
            return self._forward_generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            return self._forward_discrminator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                speech=speech,
                speech_lengths=speech_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )

    def _forward_generator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform generator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, dur_nll, _, start_idxs, _, z_mask, outs_ = outs
        _, z_p, m_p, logs_p, _, logs_q = outs_
        speech_ = get_segments(
                x=speech,
                start_idxs=start_idxs * self.generator.upsample_factor,
                segment_size=self.generator.segment_size * self.generator.upsample_factor,
            )
        if self.use_pqmf:
            speech_hat_sub_ = speech_hat_
            speech_hat_ = self.pqmf.synthesis(speech_hat_)
            speech_ = get_segments(
                x=speech,
                start_idxs=start_idxs * self.generator.upsample_factor * 4,
                segment_size=self.generator.segment_size * self.generator.upsample_factor * 4,
            )
            speech_sub_ = self.pqmf.analysis(speech_)
            

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_)
        with torch.no_grad():
            # do not store discriminator gradient in generator turn
            p = self.discriminator(speech_)

        # calculate losses
        with autocast(enabled=False):
            if self.use_pqmf:
                sc_loss, mag_loss = self.stft_loss(speech_hat_, speech_)
                stft_loss = sc_loss + mag_loss
                sub_sc_loss, sub_mag_loss = self.sub_stft_loss(speech_hat_sub_, speech_sub_)
                sub_stft_loss = sub_sc_loss + sub_mag_loss
                multiband_loss = 0.5 * (stft_loss + sub_stft_loss)
                multiband_loss = multiband_loss * 5.0      # scale stft loss
            if self.use_xv_loss:
                with torch.no_grad():
                    xv_loss = 0.0
                    # for i, (waveform, xv_target) in enumerate(zip(speech_hat_, spembs)):
                    #     xv_loss_ = self.xv_loss(waveform, xv_target)
                    #     xv_loss += xv_loss_
                    for i, (waveform, waveform_target) in enumerate(zip(speech_hat_, speech_)):
                        xv_loss_ = self.xv_loss.wav_wav_loss(waveform, waveform_target)
                        xv_loss += xv_loss_
                    xv_loss /= i+1
                    xv_loss = xv_loss * 5.0      # scale xvector loss
            mel_loss = self.mel_loss(speech_hat_, speech_)
            kl_loss = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            dur_loss = torch.sum(dur_nll.float())
            adv_loss = self.generator_adv_loss(p_hat)
            feat_match_loss = self.feat_match_loss(p_hat, p)

            mel_loss = mel_loss * self.lambda_mel
            kl_loss = kl_loss * self.lambda_kl
            dur_loss = dur_loss * self.lambda_dur
            adv_loss = adv_loss * self.lambda_adv
            feat_match_loss = feat_match_loss * self.lambda_feat_match
            loss = mel_loss + kl_loss + dur_loss + adv_loss + feat_match_loss

        stats = dict(
            generator_loss=loss.item(),
            generator_mel_loss=mel_loss.item(),
            generator_kl_loss=kl_loss.item(),
            generator_dur_loss=dur_loss.item(),
            generator_adv_loss=adv_loss.item(),
            generator_feat_match_loss=feat_match_loss.item(),
        )

        if self.use_pqmf:
            loss += multiband_loss
            stats.update(generator_loss=loss.item())
            stats.update(generator_multiband_loss=multiband_loss.item())
        
        if self.use_xv_loss:
            loss += xv_loss
            stats.update(generator_loss=loss.item())
            stats.update(generator_xvector_loss=xv_loss.item())

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # needed for trainer
        }

    def _forward_discrminator(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Perform discriminator forward.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).

        Returns:
            Dict[str, Any]:
                * loss (Tensor): Loss scalar tensor.
                * stats (Dict[str, float]): Statistics to be monitored.
                * weight (Tensor): Weight tensor to summarize losses.
                * optim_idx (int): Optimizer index (0 for G and 1 for D).

        """
        # setup
        batch_size = text.size(0)
        feats = feats.transpose(1, 2)
        speech = speech.unsqueeze(1)

        # calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outs = self.generator(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
            )
        else:
            outs = self._cache

        # store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = outs

        # parse outputs
        speech_hat_, _, _, start_idxs, *_ = outs
        speech_ = get_segments(
            x=speech,
            start_idxs=start_idxs * self.generator.upsample_factor,
            segment_size=self.generator.segment_size * self.generator.upsample_factor,
        )
        if self.use_pqmf:
            speech_hat_ = self.pqmf.synthesis(speech_hat_)
            speech_ = get_segments(
                x=speech,
                start_idxs=start_idxs * self.generator.upsample_factor * 4,
                segment_size=self.generator.segment_size * self.generator.upsample_factor * 4,
            )  

        # calculate discriminator outputs
        p_hat = self.discriminator(speech_hat_.detach())
        p = self.discriminator(speech_)

        # calculate losses
        with autocast(enabled=False):
            real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
            loss = real_loss + fake_loss

        stats = dict(
            discriminator_loss=loss.item(),
            discriminator_real_loss=real_loss.item(),
            discriminator_fake_loss=fake_loss.item(),
        )
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # reset cache
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # needed for trainer
        }

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        noise_scale_dur: float = 0.8,
        alpha: float = 1.0,
        max_len: Optional[int] = None,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (T_text,).
            feats (Tensor): Feature tensor (T_feats, aux_channels).
            sids (Tensor): Speaker index tensor (1,).
            spembs (Optional[Tensor]): Speaker embedding tensor (spk_embed_dim,).
            lids (Tensor): Language index tensor (1,).
            durations (Tensor): Ground-truth duration tensor (T_text,).
            noise_scale (float): Noise scale value for flow.
            noise_scale_dur (float): Noise scale value for duration predictor.
            alpha (float): Alpha parameter to control the speed of generated speech.
            max_len (Optional[int]): Maximum length.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]:
                * wav (Tensor): Generated waveform tensor (T_wav,).
                * att_w (Tensor): Monotonic attention weight tensor (T_feats, T_text).
                * duration (Tensor): Predicted duration tensor (T_text,).

        """
        # setup
        text = text[None]
        text_lengths = torch.tensor(
            [text.size(1)],
            dtype=torch.long,
            device=text.device,
        )
        if sids is not None:
            sids = sids.view(1)
        if lids is not None:
            lids = lids.view(1)
        if durations is not None:
            durations = durations.view(1, 1, -1)

        # inference
        if use_teacher_forcing:
            assert feats is not None
            feats = feats[None].transpose(1, 2)
            feats_lengths = torch.tensor(
                [feats.size(2)],
                dtype=torch.long,
                device=feats.device,
            )
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                feats=feats,
                feats_lengths=feats_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                max_len=max_len,
                use_teacher_forcing=use_teacher_forcing,
            )
        else:
            wav, att_w, dur = self.generator.inference(
                text=text,
                text_lengths=text_lengths,
                sids=sids,
                spembs=spembs,
                lids=lids,
                dur=durations,
                noise_scale=noise_scale,
                noise_scale_dur=noise_scale_dur,
                alpha=alpha,
                max_len=max_len,
            )
        if self.use_pqmf:
            wav = self.pqmf.synthesis(wav)
        return dict(wav=wav.view(-1), att_w=att_w[0], duration=dur[0])
