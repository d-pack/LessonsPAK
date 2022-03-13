from typing import Callable, List, NamedTuple, Optional

import numpy as np
from scipy import special

Array = np.ndarray


def ensure_array(array):
    return np.array(array, dtype="float32", copy=False)


class Dependency(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[Array], Array]


class Tensor:
    def __init__(
        self,
        data,
        depends_on: Optional[List[Dependency]] = None,
        requires_grad: bool = False,
    ) -> None:
        self.data = ensure_array(data)
        self.depends_on = depends_on or []
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None

    def __sub__(self, other) -> "Tensor":
        return sub(self, other)

    def __mul__(self, other) -> "Tensor":
        return mul(self, other)

    def __pow__(self, other) -> "Tensor":
        return power(self, other)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def sum(self) -> "Tensor":
        return reduce_sum(self)

    def sigmoid(self) -> "Tensor":
        return sigmoid(self)

    def zero_grad_(self) -> None:
        # TODO
        raise NotImplementedError

    def tolist(self):
        return self.data.tolist()

    @property
    def shape(self):
        return self.data.shape

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        if grad is None:
            if np.prod(self.data.shape) == 1:
                grad = Tensor(1)
            else:
                raise RuntimeError

        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))

        raise NotImplementedError
        # TODO
        # hint: 4 lines of code


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def reduce_sum(inp: Tensor) -> Tensor:
    # TODO
    data = ...

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data, depends_on=depends_on, requires_grad=requires_grad)


def sub(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data = ...

    depends_on = []
    if left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=left, grad_fn=grad_fn_left))

    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def mul(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data = ...

    depends_on = []
    if left.requires_grad:

        def grad_fn_left(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=left, grad_fn=grad_fn_left))

    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def power(inp: Tensor, exponent: int) -> Tensor:
    # TODO
    data = ...

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def sigmoid(inp: Tensor) -> Tensor:
    # TODO
    data = ...

    requires_grad = inp.requires_grad

    depends_on = []
    if requires_grad:

        def grad_fn(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=inp, grad_fn=grad_fn))

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


def matmul(left: Tensor, right: Tensor) -> Tensor:
    # TODO
    data = ...

    depends_on = []
    if right.requires_grad:

        def grad_fn_right(grad: Array) -> Array:
            # TODO
            raise NotImplementedError

        depends_on.append(Dependency(tensor=right, grad_fn=grad_fn_right))

    requires_grad = left.requires_grad or right.requires_grad

    return Tensor(data=data, depends_on=depends_on, requires_grad=requires_grad)


class SGD:
    def __init__(self, parameters: list, lr: float = 1e-3) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        # TODO
        raise NotImplementedError

    def zero_grad(self):
        # TODO
        raise NotImplementedError


def mse_loss(inp: Tensor, target: Tensor) -> Tensor:
    # TODO
    raise NotImplementedError
