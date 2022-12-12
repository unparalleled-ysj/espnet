import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from scipy.signal import get_window
from espnet2.gan_tts.other_vocoder.stft import STFT, TorchSTFT
import math
import onnxruntime


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


class Multiband_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=0):
        super(Multiband_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []
 
        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands*(self.post_n_fft + 2), 7, 1, padding=3))
        
        self.subband_conv_post.apply(init_weights)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.stft = STFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft)

    def forward(self, x, g=None):
      x = self.conv_pre(x) #[B, ch, length]

      if g is not None:
          x = x + self.cond(g)
        
      for i in range(self.num_upsamples):
          x = F.leaky_relu(x, LRELU_SLOPE)
          x = self.ups[i](x)
          
          
          xs = None
          for j in range(self.num_kernels):
              if xs is None:
                  xs = self.resblocks[i*self.num_kernels+j](x)
              else:
                  xs += self.resblocks[i*self.num_kernels+j](x)
          x = xs / self.num_kernels
          
      x = F.leaky_relu(x)
      x = self.reflection_pad(x)
      x = self.subband_conv_post(x)
      x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1]))

      spec = torch.exp(x[:,:,:self.post_n_fft // 2 + 1, :])
      phase = math.pi*torch.sin(x[:,:, self.post_n_fft // 2 + 1:, :])

      y_mb_hat = self.stft.inverse(torch.reshape(spec, (spec.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1])), torch.reshape(phase, (phase.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1])))
      y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
      y_mb_hat = y_mb_hat.squeeze(-2)

      return y_mb_hat

    def remove_weight_norm(self):
      print('Removing weight norm...')
      for l in self.ups:
          remove_weight_norm(l)
      for l in self.resblocks:
          l.remove_weight_norm()


class Multistream_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=0):
        super(Multistream_iSTFT_Generator, self).__init__()
        # self.h = h
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.ups.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.reshape_pixelshuffle = []
 
        self.subband_conv_post = weight_norm(Conv1d(ch, self.subbands*(self.post_n_fft + 2), 7, 1, padding=3))
        
        self.subband_conv_post.apply(init_weights)
        
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.stft = STFT(filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft)

        updown_filter = torch.zeros((self.subbands, self.subbands, self.subbands)).float()
        for k in range(self.subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.multistream_conv_post = weight_norm(Conv1d(4, 1, kernel_size=63, bias=False, padding=get_padding(63, 1)))
        self.multistream_conv_post.apply(init_weights)

        if gin_channels > 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
        
    def forward(self, x, g=None):

      x = self.conv_pre(x)#[B, ch, length]

      if g is not None:
          x = x + self.cond(g)
        
      for i in range(self.num_upsamples):

          
          x = F.leaky_relu(x, LRELU_SLOPE)
          x = self.ups[i](x)
          
          
          xs = None
          for j in range(self.num_kernels):
              if xs is None:
                  xs = self.resblocks[i*self.num_kernels+j](x)
              else:
                  xs += self.resblocks[i*self.num_kernels+j](x)
          x = xs / self.num_kernels
          
      x = F.leaky_relu(x)
      x = self.reflection_pad(x)
      x = self.subband_conv_post(x)
      x = torch.reshape(x, (x.shape[0], self.subbands, x.shape[1]//self.subbands, x.shape[-1]))

      spec = torch.exp(x[:,:,:self.post_n_fft // 2 + 1, :])
      phase = math.pi*torch.sin(x[:,:, self.post_n_fft // 2 + 1:, :])

      magnitude = torch.reshape(spec, (spec.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, spec.shape[-1]))
      phase = torch.reshape(phase, (phase.shape[0]*self.subbands, self.gen_istft_n_fft // 2 + 1, phase.shape[-1]))

      y_mb_hat = self.stft.inverse(magnitude, phase)
      y_mb_hat = torch.reshape(y_mb_hat, (x.shape[0], self.subbands, 1, y_mb_hat.shape[-1]))
      y_mb_hat = y_mb_hat.squeeze(-2)

      y_mb_hat = F.conv_transpose1d(y_mb_hat, self.updown_filter.to(x.device) * self.subbands, stride=self.subbands)

      y_g_hat = self.multistream_conv_post(y_mb_hat)

      return y_g_hat

    def remove_weight_norm(self):
      print('Removing weight norm...')
      for l in self.ups:
          remove_weight_norm(l)
      for l in self.resblocks:
          l.remove_weight_norm()


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)
