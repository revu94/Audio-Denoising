import gc

import numpy as np
import torch
from pesq import pesq
from scipy import interpolate
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

writer = SummaryWriter("runs/experiment_1")
writer = SummaryWriter("runs/metrics_with_error")

from models.metrics import AudioMetrics, AudioMetrics2

base_path = "/content/gdrive/MyDrive/Colab Notebooks/Noise2Noise"
train_on_gpu = torch.cuda.is_available()
DEVICE = torch.device("cuda" if train_on_gpu else "cpu")


def test_set_metrics(
    test_loader, model, n_fft=(48000 * 64) // 1000, hop_length=(48000 * 16) // 1000
):
    metric_names = ["CSIG", "CBAK", "COVL", "PESQ", "SSNR", "STOI"]
    overall_metrics = [[] for i in range(len(metric_names))]

    for i, (noisy, clean) in enumerate(test_loader):
        x_est = model(noisy.to(DEVICE, is_istft=True))
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
                x_est = net(noisy.to(DEVICE), is_istft=True)
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
        writer.add_scalar("Metric/mean", np.mean(overall_metrics[i]), i)
        writer.add_scalar(
            "Metric/upper_bound",
            np.mean(overall_metrics[i]) + np.std(overall_metrics[i]),
            i,
        )
        writer.add_scalar(
            "Metric/lower_bound",
            np.mean(overall_metrics[i]) - np.std(overall_metrics[i]),
            i,
        )
        writer.add_scalars(
            "Metric",
            {
                "mean": np.mean(overall_metrics[i]),
                "upper_bound": np.mean(overall_metrics[i]) + np.std(overall_metrics[i]),
                "lower_bound": np.mean(overall_metrics[i]) - np.std(overall_metrics[i]),
            },
            i,
        )
        print(
            "{} : {:.3f}+/-{:.3f}".format(
                metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])
            )
        )
    return results


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
    epochs = 20
    testmet_losses = []
    val_loss_per_epoch = []
    train_loss_per_epoch = []
    for epoch in (pbar := tqdm(range(epochs))):
        c = 0
        train_loss = 0
        n2n.train()
        for input, target in train_dataloader:
            input, target = input.to(DEVICE), target.to(DEVICE)

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
        writer.add_scalar("Loss/train", train_loss, epoch)
        scheduler.step()
        train_loss_per_epoch.append(train_loss)

        with torch.no_grad():
            val_loss, testmet = test_epoch(
                n2n, val_dataloader, loss_function, n_fft, hop_length, use_net=True
            )
        val_loss_per_epoch.append(val_loss)
        testmet_losses.append(testmet)
        torch.save(
            n2n.state_dict(),
            base_path + "/Weights/dc20_model_" + str(epoch + 1) + ".pth",
        )
        torch.save(
            optimizer.state_dict(),
            base_path + "/Weights/dc20_opt_" + str(epoch + 1) + ".pth",
        )

        pbar.set_postfix({"train_loss": float(train_loss)})
        # pbar.set_postfix({"val_loss": float(val_loss)})
    return train_loss_per_epoch, val_loss_per_epoch


def test_epoch(net, test_loader, loss_fn, n_fft, hop_length, use_net=True):
    net.eval()
    test_ep_loss = 0.0
    counter = 0.0
    # for noisy_x, clean_x in test_loader:
    #     # get the output from the model
    #     noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)
    #     pred_x = net(noisy_x)

    #     # calculate loss
    #     loss = loss_fn(
    #         noisy_x,
    #         pred_x,
    #         clean_x,
    #         n_fft,
    #         hop_length,
    #     )
    #     # Calc the metrics here
    #     test_ep_loss += loss.item()

    #     counter += 1

    # test_ep_loss /= counter

    # print("Actual compute done...testing now")

    testmet = getMetricsonLoader(test_loader, net, use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss, testmet
