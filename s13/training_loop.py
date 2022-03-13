import gzip
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import tqdm

import tor4

ROOT = Path(os.path.abspath(__file__)).parent / "mnist"
BASE_URL = "http://yann.lecun.com/exdb/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist():
    for m in MNIST_FILES:
        if (ROOT / m).exists():
            continue

        r = requests.get(f"{BASE_URL}{m}")
        with open(ROOT / m, "wb") as f:
            f.write(r.content)


def get_mnist():
    if not ROOT.exists():
        ROOT.mkdir()
        download_mnist()

    out = []
    for m in MNIST_FILES:
        with gzip.open(ROOT / m, "rb") as f:
            f.read(8)
            if "images" in m:
                f.read(8)

            arr = np.frombuffer(f.read(), dtype="uint8")
            if "images" in m:
                arr = arr.reshape((-1, 1, 28, 28))

            out.append(arr)

    return out


def transform(x):
    """Normalize images. Divide on 255, flatten n-d array to 1-d array and pad with ones"""
    # x: [b, c, h, w]

    # TODO
    raise NotImplementedError


def get_dataloader(X, y, batch_size=32, shuffle=False):
    n = len(X)
    inds = np.arange(n)

    if shuffle:
        np.random.shuffle(inds)

    for _ in range(1 if shuffle else 1):
        for start in range(0, n, batch_size):
            end = start + batch_size
            sl = inds[start:end]

            yield tor4.tensor(X[sl]), tor4.tensor(y[sl])


def ohe(y):
    """Perform One Hot Encoding"""

    # TODO
    raise NotImplementedError


class LinearModel:
    def __init__(self, in_features, n_classes):
        """Linear Model without explicit bias. Weights are initialized according to normal distribution"""

        # TODO
        self.weight = ...


    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        """Returns list of all trainable parameters"""

        # TODO
        raise NotImplementedError

    def forward(self, x):
        """Performs forward pass"""
        # x: [b, f]

        # TODO
        raise NotImplementedError


def get_model(in_features=28 * 28, n_classes=10):
    model = LinearModel(in_features=in_features, n_classes=n_classes)

    return model


def get_criterion():
    def criterion(logits, labels):
        """Calculate mean sqared error between sigmoids and labels"""

        # TODO
        raise NotImplementedError

    return criterion


def get_optimizer(params, lr=1e-3):
    opt = tor4.SGD(params, lr=lr)

    return opt


def epoch_step(loader, model, criterion, opt=None, desc=""):
    is_train = opt is not None

    with tqdm.tqdm(loader, desc=desc, mininterval=2, leave=False) as pbar:
        loc_loss = loc_acc = n = 0
        for x, y in pbar:
            bs = x.shape[0]

            # TODO
            # perform forward pass and calculate loss
            logits = ...
            loss = ...

            if is_train:
                # TODO
                # 3 magic words
                ...

            loc_loss += loss.data * bs
            loc_acc += (logits.data.argmax(-1) == y.data.argmax(-1)).sum()
            n += bs

            pbar.set_postfix(
                **{
                    "loss": f"{loc_loss / n:.6}",
                    "acc": f"{loc_acc / n:.6}",
                }
            )

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
        ax.grid(ls="--")

    fig.savefig("learning_curve")
    plt.close(fig)


def main():
    # Training/test sets
    X_train, y_train, X_test, y_test = get_mnist()
    # import pdb; pdb.set_trace()
    X_train = transform(X_train)
    X_test = transform(X_test)

    y_train = ohe(y_train).astype("float32")
    y_test = ohe(y_test).astype("float32")

    # Model
    model = get_model()
    criterion = get_criterion()

    opt = get_optimizer(model.parameters())

    # Hyperparameters
    n_epochs = 100

    # History
    history = defaultdict(lambda: defaultdict(list))

    # Training
    for epoch in range(n_epochs):
        loss, acc = epoch_step(
            get_dataloader(X_train, y_train, shuffle=True),
            model,
            criterion,
            opt=opt,
            desc=f"[ Training {epoch}/{n_epochs}]",
        )
        history["loss"]["train"].append(loss)
        history["accuracy"]["train"].append(acc)

        # Validation
        loss, acc = epoch_step(
            get_dataloader(X_test, y_test),
            model,
            criterion,
            opt=None,
            desc=f"[ Validation {epoch}/{n_epochs}]",
        )
        history["loss"]["dev"].append(loss)
        history["accuracy"]["dev"].append(acc)

        plot(history)


if __name__ == "__main__":
    main()
