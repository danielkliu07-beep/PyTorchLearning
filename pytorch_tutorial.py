import torch

#Tensor patterns:

#1 - Turn a python list into a tensor
data = [[1, 2, 3], [4, 5, 6]]
my_tensor = torch.tensor(data)

print(my_tensor)

#2 - How to initialize model weights if you know shape, but not the values (starting point for a model)
shape = (2, 3) #shape tuple (2 rows, 3 columns)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
random = torch.randn(shape)

print(f"Random Tensor:\n {random}")

#3 - Creating a tensor with the same shape and type as another tensor
template = torch.tensor([[1, 2], [3, 4]])
rand_like = torch.randn_like(template, dtype = torch.float) #data type = torch.float

print(f"Template Tensor:\n {template}\n")
print(f"Randn_like Tensor:\n {rand_like}")


#Tensor attribute:

tensor = torch.randn(2, 3)
print(f"Shape: {tensor.shape}")
print(f"Datatype: {tensor.dtype}")
print(f"Device: {tensor.device}")

#.shape - describes the dimensions of the tensors. The #1 debugging tool as 90% of errors in Pytorch will be shape mismatches
#.device - where the tensor lives. Either the cpu or cuda (gpu)
#.dtype - the data type of the numbers. Default is float32


#Rules for models:

#Model parameters like weights or biases must be a float type. float32 is standard.
#Data that represents categories or counts can be integers


#Autograd:

#requires_grad = True - turns on the autograd engine
x_data = torch.tensor([[1., 2.], [3., 4.]]) #Standard data tensor
w = torch.tensor([[1.0], [2.0]], requires_grad = True) #Parameter tensor

print(f"Data tensor requires_grad: {x_data.requires_grad}")
print(f"Parameter tensor requires_grad: {w.requires_grad}")









