from collections import defaultdict

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt


def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform


def get_dataset(transform=None):
    train_ds = torchvision.datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = torchvision.datasets.MNIST(
        root='./data/',
        train=False,
        transform=transform,
    )

    return train_ds, test_ds


def get_dataloader(train_ds, test_ds, batch_size=32, num_workers=0):
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dl, test_dl


class LinearModel(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(n_classes, in_features),
            requires_grad=True,
        )
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)

    def forward(self, x):
        # x: [b, c, h, w]

        bs = x.size(0)
        x = x.view(bs, -1)  # [b, c*h*w]

        x = torch.nn.functional.linear(x, self.weight, self.bias)  # [b, n_classes]

        return x


def get_model(in_features=28*28, n_classes=10):
    model = LinearModel(in_features=in_features, n_classes=n_classes)

    return model


def get_criterion():
    criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(params, lr=1e-3):
    opt = torch.optim.SGD(params, lr=lr)

    return opt


def epoch_step(loader, model, criterion, opt=None, desc=''):
    is_train = opt is not None
    if is_train:
        model.train()
        criterion.train()
    else:
        model.eval()
        criterion.eval()

    with tqdm.tqdm(loader, desc=desc, mininterval=2, leave=False) as pbar:
        loc_loss = loc_acc = n = 0
        for x, y in pbar:
            if torch.cuda.is_available():
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            bs = len(x)

            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loc_loss += loss.item() * bs
            loc_acc += (logits.argmax(dim=-1) == y).sum().item()
            n += bs

            pbar.set_postfix(**{
                'loss': f'{loc_loss / n:.6}',
                'acc': f'{loc_acc / n:.6}',
            })

    return loc_loss / n, loc_acc / n


def plot(history):
    fig: plt.Figure = plt.figure(figsize=(16, 4))
    for i, metric in enumerate(history, start=1):
        ax: plt.axes.Axis = fig.add_subplot(1, 2, i)
        for dataset in history[metric]:
            ax.plot(
                history[metric][dataset],
                label=f"{dataset} ({history[metric][dataset][-1]:.3f})",
            )

        ax.set_xlabel("#epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(ls='--')

    fig.savefig('learning_curve')
    plt.close(fig)


def main():
    # Training/test sets

    transform = get_transform()
    train_ds, test_ds = get_dataset(transform)
    train_dl, test_dl = get_dataloader(train_ds, test_ds)
    # x, _ = next(iter(train_dl))
    # images = torchvision.utils.make_grid(x)
    # import pdb; pdb.set_trace()

    # Model
    model = get_model()
    criterion = get_criterion()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    opt = get_optimizer(model.parameters())

    # Hyperparameters
    n_epochs = 100

    # History
    best_acc = 0
    history = defaultdict(lambda: defaultdict(list))

    # Save model weights
    saver = lambda path: torch.save({
        "state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "history": dict(history),
        # ...
    }, path)

    # Training
    for epoch in range(n_epochs):
        loss, acc = epoch_step(train_dl, model, criterion, opt=opt,
            desc=f'[ Training {epoch}/{n_epochs}]',
        )
        history['loss']['train'].append(loss)
        history['accuracy']['train'].append(acc)

        # Validation
        with torch.no_grad():  # <- no backprop, hence no need gradients
            loss, acc = epoch_step(test_dl, model, criterion, opt=None,
                desc=f'[ Validation {epoch}/{n_epochs}]',
            )
            history['loss']['dev'].append(loss)
            history['accuracy']['dev'].append(acc)

            if acc > best_acc:
                best_acc = acc
                saver('best.pth')

        plot(history)
        saver('last.pth')


if __name__ == '__main__':
    main()
