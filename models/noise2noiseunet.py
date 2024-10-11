import gc

import torch
from torch import nn

from models.unet_model import Decoder, Encoder


class Noise2NoiseUNet(nn.Module):
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.encoders = [
            Encoder(1, 32, (7, 1), (1, 1), (3, 0)),
            Encoder(32, 32, (1, 7), (1, 1), (0, 3)),
            Encoder(32, 64, (6, 4), (2, 2), (0, 0)),
            Encoder(64, 64, (7, 5), (2, 1), (0, 0)),
            Encoder(64, 64, (5, 3), (2, 2), (0, 0)),
            Encoder(64, 64, (5, 3), (2, 1), (0, 0)),
            Encoder(64, 64, (5, 3), (2, 2), (0, 0)),
            Encoder(64, 64, (5, 3), (2, 1), (0, 0)),
            Encoder(64, 64, (5, 3), (2, 2), (0, 0)),
            Encoder(64, 128, (5, 3), (2, 1), (0, 0)),
        ]

        self.decoders = [
            Decoder(128, 64, (6, 3), (2, 1), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 2), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 1), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (6, 4), (2, 2), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 1), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (6, 4), (2, 2), (0, 0), (0, 0)),
            Decoder(64 * 2, 64, (8, 5), (2, 1), (0, 0), (0, 0)),
            Decoder(128, 32, (7, 5), (2, 2), (0, 0), (0, 0)),
            Decoder(64, 32, (1, 7), (1, 1), (0, 3), (0, 0)),
            Decoder(64, 1, (7, 1), (1, 1), (3, 0), (0, 0), last_layer=True),
        ]

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.model_length = len(self.decoders) - 1

    def forward(self, x, is_istft=True):
        orig_x = x

        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            x = encoder(x)
            # print("Encoder : ", x.shape)

        p = x
        for i, decoder in enumerate(self.decoders):
            # print("i", i)
            p = decoder(p)
            if i < self.model_length:
                # skip connection
                p = torch.cat([p, xs[self.model_length - i]], dim=1)
            # print("Decoder : ", p.shape)
        # print("p", p)

        mask = p

        output = mask * orig_x
        output = torch.squeeze(output, 1)

        if is_istft:
            output = torch.view_as_complex(output)
            output = torch.istft(
                output,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                normalized=True,
                return_complex=False,
            )
        return output
