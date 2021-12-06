import torch
import torch.nn as nn


def print_xo(x, o):
    print(f"\n{x.shape=}")
    print(f"\n{x=}")
    print(f"\n{o.shape=}")
    print(f"\n{o=}")


def conv():
    w = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=2,
        stride=1,
        dilation=3,
        padding=1,
        bias=False,
    )
    w.weight.requires_grad_(False)

    w.weight.zero_()
    w.weight[0, :, 0, 0] = 1
    w.weight[0, :, 1, 1] = -1
    w.weight[1, :, 0, 0] = -1
    w.weight[1, :, 1, 1] = 1
    print(w.weight.shape)
    print(w.weight)

    x = torch.arange(3*3).float().view(1, 1, 3, 3)  # [b, c, h, w]
    o = w(x)

    print_xo(x, o)


def maxpool():
    w = nn.MaxPool2d(kernel_size=2, stride=1)
    x = torch.arange(3*3).float().view(1, 1, 3, 3)  # [b, c, h, w]
    o = w(x)

    print_xo(x, o)


def batchnorm():
    d = 3
    w = nn.BatchNorm1d(num_features=d)
    print(w.weight)
    print(w.bias)

    # w.eval()
    b = 2
    x = torch.arange(b*d).float().view(-1, d)  # [b, d]
    o = w(x)

    print_xo(x, o)


def dropout():
    w = nn.Dropout(p=0.5)
    w.eval()
    x = torch.arange(3).float().view(1, 3)  # [b, d]

    o = w(x)

    print_xo(x, o)


def main():
    conv()
    # maxpool()
    # batchnorm()
    # dropout()


if __name__ == "__main__":
    main()
