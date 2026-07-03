import torch
import torch.nn as nn
import torch.optim as optim

#Model blueprint:
#__init__ - define layers
#forward - connect layers

class LinearRegressionModel(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear_layer = nn.Linear(in_features, out_features)
    
    def forward(self, x):

        return self.linear_layer(x)

model = LinearRegressionModel(in_features = 1, out_features = 1)

print("Model Architecture")
print(model)

learning_rate = 0.01

optimizer = optim.Adam(model.parameters(), lr = learning_rate) #model.parameters() - Optimize the parameters

loss_fn = nn.MSELoss() #Mean squared error loss

N = 10 
D_in = 1
D_out = 1
X = torch.randn(N, D_in)
true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 


epochs = 100

for epoch in range(epochs):

    y_hat = model(X)

    loss = loss_fn(y_hat, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss = {loss.item():.4f}")


