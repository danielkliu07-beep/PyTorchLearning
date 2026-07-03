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



#Autograd in action

a = torch.tensor(2.0, requires_grad = True)
b = torch.tensor(3.0, requires_grad = True)
x = torch.tensor(4.0, requires_grad = True)

y = a + b

z = x * y

print(f"Result z: {z}")

#.grad_fn points to the function that created the tensor
print(f"grad_fn for z: {z.grad_fn}")
print(f"grad_fn for y: {y.grad_fn}")
print(f"grad_fn for a: {a.grad_fn}")



#Operations

#1 - '*': (element-wise multiplication) multiplies matching positions
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[10, 20], [30, 40]])

element_wise_product = a * b

print(element_wise_product)

#2 - '@': matrix multiplication
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
m2 = torch.tensor([[7, 8], [9, 10], [11, 12]])

matrix_product = m1 @ m2

print(matrix_product)

#3 - 'sum(), mean(), max()': reduction operations
scores = torch.tensor([[10., 20., 30.], [5., 10., 15.]]) #2 students by 3 assignments

average_score = scores.mean()

print(f"Overall Mean: {average_score}")

#4 - dim argument 
scores = torch.tensor([[10., 20., 30.], [5., 10., 15.]])

avg_per_assignment = scores.mean(dim = 0) #dim = 0: collapses the rows, operates vertically
avg_per_student = scores.mean(dim = 1) #dim = 1: collapses the columns, operates horizontally

print(f"Average per assignment (dim = 0): {avg_per_assignment}")
print(f"Average per student (dim = 1): {avg_per_student}")

#5 - Basic Indexing
x = torch.arange(12).reshape(3, 4)
col2 = x[:, 2] #3rd column of the matrix (index 2)

print(x)
print(col2)

#6 - Argmax: finds the index of the highest value (used to find a model's final prediction)
scores = torch.tensor([
    [10, 0, 5, 20, 1], #best score is at index 3
    [1, 30, 2, 5, 0] #best score is at index 1
])

best_indices = torch.argmax(scores, dim = 1) #dim = 1 -> looks only at the individual rows

print(best_indices)

#7 - torch.gather() - finds the specific elements at a row or column
data = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]
])

indices_to_select = torch.tensor([[2], [0], [3]]) #list of columns to get from each row

selected_values = torch.gather(data, dim = 1, index = indices_to_select)

print(selected_values)







