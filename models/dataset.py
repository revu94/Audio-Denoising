import numpy as np
import torch
import torchaudio  # type: ignore
from torch.utils.data import Dataset


class N2NDataset(Dataset):
    def __init__(self, input, target, n_fft=64, hop_length=16):
        super().__init__()
        self.input_files = sorted(input)
        self.output_files = sorted(target)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.len = len(self.input_files)
        self.max_len = 165000

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_target, _ = torchaudio.load(self.output_files[index])
        x_input, _ = torchaudio.load(self.input_files[index])
        x_target = self.sample(x_target)
        x_input = self.sample(x_input)

        x_input_stft = torch.stft(
            input=x_input,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            return_complex=False,
        )
        x_target_stft = torch.stft(
            input=x_target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            return_complex=False,
        )
        return x_input_stft, x_target_stft

    def sample(self, waveform):
        waveform = waveform.numpy()
        current_len = waveform.shape[1]
        output = np.zeros((1, self.max_len), dtype="float32")
        output[0, -current_len:] = waveform[0, : self.max_len]
        output = torch.from_numpy(output)
        return output
