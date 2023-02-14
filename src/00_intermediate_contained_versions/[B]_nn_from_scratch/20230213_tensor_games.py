import torch
x = torch.tensor([[1,2]])
y = torch.tensor([[1],[2]])

print(x.shape)
# torch.Size([1,2]) # one entity of two items
print(y.shape)
# torch.Size([2,1]) # two entities of one item each
print(x.dtype)
# torch.int64

x = torch.tensor([False, 1, 2.0])
print(x)
# tensor([0., 1., 2.])

torch.ones((3, 4))

torch.randint(low=0, high=10, size=(3,4))


import torch
x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
print(x)

out = x.pow(2).sum()

out.backward()

x.grad
print(x)