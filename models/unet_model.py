import torch
from torch import nn


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        self.im_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)

        output = torch.stack([c_real, c_im], dim=-1)
        print(output.shape)
        return output


class ConvT2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal, stride, padding=0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal = kernal
        self.stride = stride
        self.padding = padding

        self.conv2d_real = nn.ConvTranspose2d(
            in_channels, out_channels, kernal, stride, padding
        )
        self.conv2d_im = nn.ConvTranspose2d(
            in_channels, out_channels, kernal, stride, padding
        )
        nn.init.xavier_uniform_(self.conv2d_real.weight)
        nn.init.xavier_uniform_(self.conv2d_im.weight)

    def forward(self, x):
        x_conv2d_real = x[..., 0]
        x_conv2d_im = x[..., 1]

        output_conv2d_real = self.conv2d_real(x_conv2d_real) - self.conv2d_im(
            x_conv2d_im
        )
        output_conv2d_im = self.conv2d_real(x_conv2d_im) + self.conv2d_im(x_conv2d_real)
        output_conv2d = torch.stack([output_conv2d_real, output_conv2d_im], -1)

        return output_conv2d


class CConvTranspose2d(nn.Module):
    """
    Class of complex valued dilation convolutional layer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.real_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        self.im_convt = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)

        output = torch.stack([ct_real, ct_im], dim=-1)
        return output


class ComplexBatchNorm(nn.Module):
    def __init__(
        self,
        out_channels,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.batchnorm_real = nn.BatchNorm2d(
            out_channels,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )
        self.batchnorm_im = nn.BatchNorm2d(
            out_channels,
            eps=self.eps,
            momentum=self.momentum,
            affine=self.affine,
            track_running_stats=self.track_running_stats,
        )

    def forward(self, x):
        # print(x)
        batchnorm_x_real = x[..., 0]
        batchnorm_x_im = x[..., 1]
        batchnorm_out_real = self.batchnorm_real(batchnorm_x_real)
        batchnorm_out_im = self.batchnorm_im(batchnorm_x_im)
        batchnorm_output = torch.stack([batchnorm_out_real, batchnorm_out_im], -1)
        return batchnorm_output


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernal, stride, padding=(0, 0)):
        super().__init__()

        self.encoder = CConv2d(in_channels, out_channels, kernal, stride, padding)

        self.normalize = ComplexBatchNorm(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        output_conv = self.encoder(x)
        output_normalize = self.normalize(output_conv)
        output_activation = self.activation(output_normalize)
        return output_activation


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernal,
        stride,
        padding,
        last_layer=False,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.decoder = ConvT2dBlock(
            in_channels,
            out_channels,
            kernal,
            stride,
            padding,
        )
        self.normalize = ComplexBatchNorm(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        decoder_conv = self.decoder(x)
        if not self.last_layer:
            decoder_normed = self.normalize(decoder_conv)
            output = self.activation(decoder_normed)
        else:
            m_phase = decoder_conv / (torch.abs(decoder_conv) + 1e-8)
            m_mag = torch.tanh(torch.abs(decoder_conv))
            output = m_phase * m_mag
        return output
