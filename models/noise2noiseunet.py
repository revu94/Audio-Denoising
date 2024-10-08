import torch
from torch import nn
from tqdm.auto import tqdm

from models.unet_model import Decoder, Encoder


def loss_function(x_, y_pred_, y_true_, n_fft, hop_length, eps=1e-8):

    y_true_ = torch.squeeze(y_true_, 1)
    y_true_ = torch.view_as_complex(y_true_)
    y_true = torch.istft(y_true_, n_fft=n_fft, hop_length=hop_length, normalized=True)
    x_ = torch.squeeze(x_, 1)
    x_ = torch.view_as_complex(x_)
    x = torch.istft(x_, n_fft=n_fft, hop_length=hop_length, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (
        torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps
    )
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


def train(net, train_dataloader, optimizer, n_fft, hop_length):
    epochs = 10
    loss_per_epoch = []
    net.train()
    for epoch in (pbar := tqdm(range(epochs))):
        for input, target in train_dataloader:
            # input, target = input.to(DEVICE), target.to(DEVICE)
            output = net(input)
            loss = loss_function(
                input,
                output,
                target,
                n_fft,
                hop_length,
            )

            optimizer.zero_grad()
            loss.backward()
            loss_per_epoch.append(loss.item())
            optimizer.step()
            pbar.set_postfix({"loss": float(loss)})
    return loss_per_epoch


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
            Decoder(128, 64, (6, 3), (2, 1), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 2), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 1), (0, 0)),
            Decoder(64 * 2, 64, (6, 4), (2, 2), (0, 0)),
            Decoder(64 * 2, 64, (6, 3), (2, 1), (0, 0)),
            Decoder(64 * 2, 64, (6, 4), (2, 2), (0, 0)),
            Decoder(64 * 2, 64, (8, 5), (2, 1), (0, 0)),
            Decoder(64 * 2, 32, (7, 5), (2, 2), (0, 0)),
            Decoder(64 * 2, 32, (1, 7), (1, 1), (0, 3)),
            Decoder(32, 1, (7, 1), (1, 1), (3, 0), last_layer=True),
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
            print("i", i)
            p = decoder(p)
            if i < self.model_length:
                # skip connection
                p = torch.cat([p, xs[self.model_length - i]], dim=1)
            print("Decoder : ", p.shape)
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
