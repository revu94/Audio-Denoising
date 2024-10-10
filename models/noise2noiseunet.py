import gc

import numpy as np
import torch
from pesq import pesq
from scipy import interpolate
from torch import nn
from tqdm.auto import tqdm

from metrics import AudioMetricException, AudioMetrics, AudioMetrics2
from models.unet_model import Decoder, Encoder

train_on_gpu = torch.cuda.is_available()
DEVICE = torch.device("cuda" if train_on_gpu else "cpu")


def test_set_metrics(
    test_loader, model, n_fft=(48000 * 64) // 1000, hop_length=(48000 * 16) // 1000
):
    metric_names = ["CSIG", "CBAK", "COVL", "PESQ", "SSNR", "STOI"]
    overall_metrics = [[] for i in range(len(metric_names))]

    for i, (noisy, clean) in enumerate(test_loader):
        x_est = model(noisy, is_istft=True)
        x_est_np = x_est[0].view(-1).detach().cpu().numpy()
        # 3/10
        clean = torch.view_as_complex(clean)
        x_c_np = (
            torch.istft(
                torch.squeeze(clean[0], 1),
                n_fft=n_fft,
                hop_length=hop_length,
                normalized=True,
            )
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )
        metrics = AudioMetrics(x_c_np, x_est_np, 48000)

        overall_metrics[0].append(metrics.CSIG)
        overall_metrics[1].append(metrics.CBAK)
        overall_metrics[2].append(metrics.COVL)
        overall_metrics[3].append(metrics.PESQ)
        overall_metrics[4].append(metrics.SSNR)
        overall_metrics[5].append(metrics.STOI)

    metrics_dict = dict()
    for i in range(len(metric_names)):
        metrics_dict[metric_names[i]] = {
            "mean": np.mean(overall_metrics[i]),
            "std_dev": np.std(overall_metrics[i]),
        }

    return metrics_dict


def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old = np.linspace(0, duration, original.shape[0])
        time_new = np.linspace(
            0, duration, int(original.shape[0] * new_rate / old_rate)
        )
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


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


wonky_samples = []


def getMetricsonLoader(
    loader,
    net,
    use_net=True,
    n_fft=(48000 * 64) // 1000,
    hop_length=(48000 * 16) // 1000,
):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB", "PESQ-NB", "SNR", "SSNR", "STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i + 1) % 10 == 0:
            end_str = "\n"
        else:
            end_str = ","
        # print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            if use_net:  # Forward of net returns the istft version
                x_est = net(noisy, is_istft=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                noisy = torch.view_as_complex(noisy)
                x_est_np = (
                    torch.istft(
                        torch.squeeze(noisy, 1),
                        n_fft=n_fft,
                        hop_length=hop_length,
                        normalized=True,
                    )
                    .view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            clean = torch.view_as_complex(clean)
            x_clean_np = (
                torch.istft(
                    torch.squeeze(clean, 1),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    normalized=True,
                )
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )

            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)

            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, "wb")

            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, "nb")

            # print(new_scores)
            # print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] = np.mean(overall_metrics[i])
        temp["STD"] = np.std(overall_metrics[i])
        temp["Min"] = min(overall_metrics[i])
        temp["Max"] = max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data", addon)
    for i in range(5):
        print(
            "{} : {:.3f}+/-{:.3f}".format(
                metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])
            )
        )
    return results


def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.0
    counter = 0
    for noisy_x, clean_x in train_loader:

        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)

        # zero  gradients
        net.zero_grad()

        # get the output from the model
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        loss.backward()
        optimizer.step()

        train_ep_loss += loss.item()
        counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


def train(
    n2n,
    train_dataloader,
    val_dataloader,
    loss_function,
    optimizer,
    n_fft,
    hop_length,
    scheduler,
):
    epochs = 4
    val_losses = []
    train_loss_per_epoch = 0
    # net.train()
    for epoch in (pbar := tqdm(range(epochs))):
        n2n.train()
        train_loss = 0
        val_loss_per_epoch = []
        train_loss_per_epoch = []
        c = 0
        for input, target in train_dataloader:
            n2n.zero_grad()
            output = n2n(input)
            loss = loss_function(
                input,
                output,
                target,
                n_fft,
                hop_length,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            c += 1
        train_loss /= c
        scheduler.step()
        train_loss_per_epoch.append(train_loss)

        with torch.no_grad():
            val_loss, testmet = test_epoch(
                n2n, val_dataloader, loss_function, use_net=True
            )
        val_loss_per_epoch.append(val_loss)
        torch.save(
            n2n.state_dict(),
            "../Weights/dc20_model_" + str(epoch + 1) + ".pth",
        )
        torch.save(
            optimizer.state_dict(),
            "../Weights/dc20_opt_" + str(epoch + 1) + ".pth",
        )

        pbar.set_postfix({"train_loss": float(train_loss)})
        pbar.set_postfix({"val_loss": float(val_loss)})
    return train_loss_per_epoch, val_loss_per_epoch


def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.0
    counter = 0.0

    for noisy_x, clean_x in test_loader:
        # get the output from the model
        noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
        pred_x = net(noisy_x)

        # calculate loss
        loss = loss_fn(noisy_x, pred_x, clean_x)
        # Calc the metrics here
        test_ep_loss += loss.item()

        counter += 1

    test_ep_loss /= counter

    # print("Actual compute done...testing now")

    testmet = getMetricsonLoader(test_loader, net, use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss, testmet


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
