import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hid_dim=3, A=None, B=None):
        super().__init__()

        if A is None:
            A = torch.randn(hid_dim, 2)

        self.A = nn.Parameter(A, requires_grad=True)

        if B is None:
            B = torch.randn(1, hid_dim)

        self.B = nn.Parameter(B, requires_grad=True)

    def features(self, x):
        x = nn.functional.linear(x, self.A)
        # x = x @ self.A.T

        x = x ** 2

        return x

    def linear(self, x):
        x = nn.functional.linear(x, self.B)
        # x = x @ self.B.T

        return x

    def forward(self, x):
        features = self.features(x)

        logits = self.linear(features)
        logits = logits.squeeze(1)

        return logits


def main():
    N = 1000
    x = torch.rand(N, 2)
    y = torch.prod(x, dim=1)

    hid_dim = 2
    model = Model(hid_dim=hid_dim)

    lr = 1e-2
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    criterion = nn.MSELoss()

    n_epochs = 1_000_000
    eps = 1e-9
    for i in range(1, n_epochs + 1):
        opt.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        if i % 1000 == 0:
            print(loss.item())

        if loss.item() < eps:
            break

    print("\"Learned\" parameters")
    print(model.A)
    print(model.B)

    with torch.no_grad():
        loss = criterion(model(x), y).item()

    print(f"Train error {loss}")

    x_test = torch.rand(N, 2) + 10
    y_test = torch.prod(x_test, dim=1)
    with torch.no_grad():
        loss = criterion(model(x_test), y_test).item()

    print(f"Test error {loss}")

    A = torch.zeros(3, 2)
    A[0, 0] = A[1, 1] = A[2, 0] = A[2, 1] = 1
    B = 1 / 2 * torch.ones(1, 3)
    B[0, 0] = B[0, 1] = -1 / 2
    model_true = Model(A=A, B=B)

    with torch.no_grad():
        loss = criterion(model_true(x_test), y_test).item()

    print(f"True model: {loss}")


if __name__ == "__main__":
    main()
