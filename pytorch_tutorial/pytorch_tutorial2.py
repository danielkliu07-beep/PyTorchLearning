import torch

#Simple Linear Regression Problem: y = XW + b -> y = 2x + 1

N = 10 #10 data points, each of which has an input feature and an output value
D_in = 1
D_out = 1

X = torch.randn(N, D_in) #Create our input data X

#Expected or "true" values for W and b
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 #Add a little extra noise

W = torch.randn(D_in, D_out, requires_grad = True)
b = torch.randn(1, requires_grad = True)

y_hat = X @ W + b #forward pass

print(f"Initial Weight W:\n {W}\n")
print(f"Initial Bias b:\n {b}")

#Loss function
error = y_hat - y_true
squared_error = error ** 2
loss = squared_error.mean()

print(f"Loss: {loss}")

#Backwards pass - calculates the gradients for W and b
loss.backward()

print(f"Gradient for W (dL/dW):\n {W.grad}\n")
print(f"Gradient for b (dL/db): \n {b.grad}")

#Training process
learning_rate, epochs = 0.01, 100

W, b = torch.randn(1, 1, requires_grad = True), torch.randn(1, requires_grad = True)

for epoch in range(epochs):

    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true) ** 2)

    loss.backward()

    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad
    
    W.grad.zero_() #Set the gradients to zero for the next round of learning
    b.grad.zero_()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss = {loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")

print(f"\nFinal Parameters: W = {W.item():.3f}, b = {b.item():.3f}")
print(f"True Parameters: W = 2.000, b = 1.000")



