from tor4 import mse_loss, tensor


def test_tensor_sum():
    a = tensor(data=[-1, 1, 2])
    a_sum = a.sum()

    assert a_sum.tolist() == 2
    assert not a_sum.requires_grad


def test_tensor_sum_backward():
    a = tensor(data=[-1, 1, 2.0], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 2
    assert a_sum.requires_grad
    assert a.grad.tolist() == [1, 1, 1]


def test_tensor_sum_backward2():
    a = tensor(data=[-1, 1, 2.0], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward(tensor(3))

    assert a_sum.tolist() == 2
    assert a_sum.requires_grad
    assert a.grad.tolist() == [3, 3, 3]


def test_tensor_sum3_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 8
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]


def test_tensor_sum_keepdim3_backward():
    a = tensor(data=[[[-1], [1], [2]], [[1], [2], [3.0]]], requires_grad=True)
    a_sum = a.sum()
    a_sum.backward()

    assert a_sum.tolist() == 8
    assert a_sum.requires_grad
    assert a.grad.tolist() == [[[1], [1], [1]], [[1], [1], [1]]]


def test_tensor_sub():
    a = tensor([1, 2, 3])
    b = tensor([4, 5, 6])
    apb = a - b

    assert apb.tolist() == [-3, -3, -3]
    assert not apb.requires_grad


def test_tensor_sub_backward():
    a = tensor([1, 2, 3])
    b = tensor([-1, 0, 1.0], requires_grad=True)
    apb = a - b
    apb.backward(tensor([1, 1, 1]))

    assert apb.tolist() == [2, 2, 2]
    assert not a.requires_grad
    assert b.requires_grad
    assert apb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [-1, -1, -1]


def test_tensor_rsub_backward():
    a = tensor([1, 2, 3.0], requires_grad=True)
    b = tensor([-1, 0, 1])
    apb = a - b
    apb.backward(tensor([1, 1, 2]))

    assert apb.tolist() == [2, 2, 2]
    assert a.requires_grad
    assert not b.requires_grad
    assert apb.requires_grad
    assert a.grad.tolist() == [1, 1, 2]
    assert b.grad is None


def test_tensor_mul():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[-1, 3, 1])
    amb = a * b

    assert amb.tolist() == [-1, 6, 3]
    assert not amb.requires_grad


def test_tensor_mul_backward():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[-1, 3, 1.0], requires_grad=True)
    amb = a * b
    amb.backward(tensor([1, 2, 3]))

    assert amb.tolist() == [-1, 6, 3]
    assert not a.requires_grad
    assert b.requires_grad
    assert amb.requires_grad
    assert a.grad is None
    assert b.grad.tolist() == [1, 4, 9]


def test_tensor_rmul_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    b = tensor(data=[-1, 3, 1])
    amb = a * b
    amb.backward(tensor([3, 2, 1]))

    assert amb.tolist() == [-1, 6, 3]
    assert a.requires_grad
    assert not b.requires_grad
    assert amb.requires_grad
    assert a.grad.tolist() == [-3, 6, 1]
    assert b.grad is None


def test_tensor_pow_scalar():
    a = tensor(data=[1, 2, 3])
    ap2 = a ** 2

    assert ap2.tolist() == [1, 4, 9]
    assert not ap2.requires_grad


def test_tensor_pow_scalar_backward():
    a = tensor(data=[1, 2, 3.0], requires_grad=True)
    ap2 = a ** 2
    ap2.backward(tensor(data=[1, 1, 1]))

    assert ap2.tolist() == [1, 4, 9]
    assert ap2.requires_grad
    assert a.requires_grad
    assert a.grad.tolist() == [2, 4, 6]


def test_sigmoid_backward():
    a = tensor(data=[0.0, 0, 0], requires_grad=True)
    a_sigmoid = a.sigmoid()
    a_sigmoid.backward(tensor(data=[1, 1, 1]))

    assert a_sigmoid.tolist() == [0.5, 0.5, 0.5]
    assert a_sigmoid.requires_grad
    assert a.grad.tolist() == [0.25, 0.25, 0.25]


def test_tensor_matmul():
    a = tensor(data=[1, 2, 3])
    b = tensor(data=[4, 5, 6])
    amb = a @ b

    assert amb.tolist() == 32
    assert not amb.requires_grad


def test_tensor_matmul2():
    a = tensor(data=[[1, 2, 3], [3, 2, 1]])
    b = tensor(data=[[1, 1], [1, 1], [1, 1]])
    amb = a @ b

    assert amb.tolist() == [
        [6, 6],
        [6, 6],
    ]
    assert not amb.requires_grad
    assert amb.shape == (2, 2)


def test_tensor_matmul2_backward():
    a = tensor(data=[[1, 2, 3], [3, 2, 1.0]])
    b = tensor(data=[[1, 1], [1, 1], [1, 1.0]], requires_grad=True)
    amb = a @ b
    amb.backward(tensor(data=[[1, 1], [1, 1]]))

    assert amb.tolist() == [
        [6, 6],
        [6, 6],
    ]
    assert amb.requires_grad
    assert b.grad.tolist() == [
        [4, 4],
        [4, 4],
        [4, 4],
    ]


def test_mse_backward():
    inputs = tensor(data=[1.0, 2, 3], requires_grad=True)
    targets = tensor(data=[2, 3, 2])

    mse_nn = mse_loss(inputs, targets)
    mse = ((inputs.data - targets.data) ** 2).sum()

    mse_nn.backward()

    assert mse_nn.data == mse == 3
    assert mse_nn.requires_grad
    assert inputs.grad.tolist() == [-2, -2, 2]
