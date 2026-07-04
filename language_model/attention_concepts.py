import torch
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4, 8, 2 #batch, time (tokens), channels
x = torch.randn(B,T,C)
print(x.shape) #There's a word stored in every T location

#How to 

#Version 1
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1] #(t, C) shape
        xbow[b, t] = torch.mean(xprev, 0) #mean of the 0th dimension (time)

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
# print(a)
# print(b)
# print(c)


#Version 2
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim = True)
print(wei)

xbow2 = wei @ x #(T, T) @ (B, T, C) ---> (B, T, T) @ (B, T, C) ---> (B, T, C) (Pytorch creates a batch dimension if it's not there)
print(xbow2)
print(torch.allclose(xbow, xbow2)) #Prints if xbow == xbow2



#Version 3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = -1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))


#1 hour in

