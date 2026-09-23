import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, input_size, weights):
        super(Net, self).__init__()
        self.K = nn.Parameter(torch.randn(weights, weights) * 0.1)
        mask = torch.ones_like(self.K) - torch.eye(weights)
        mask = torch.tril(torch.triu(mask, diagonal=-1), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, w, n_steps, T):

        assert w.ndim == 2, "Input tensor must be 2D"

        dt = T / n_steps
        theta = torch.zeros_like(w)
        for _ in range(n_steps):
            coupling = theta[:, None, :] - theta[:, :, None]
            theta = theta + dt * (w + torch.einsum('bij,ij->bj', torch.sin(coupling), self.mask * self.K))
       
        return theta



if __name__ == "__main__":


    input_size = 1
    weights = 5
    n_steps = 100
    b = 100
    T = 1.0
    model = Net(input_size, weights)
    model.train()  # Set the model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):

        x = torch.zeros(b, weights)  # Example input tensor with shape (batch_size, input_size)
        x[:,:input_size] = torch.randn(b, input_size)  # Example input tensor with shape (batch_size, input_size)

        y = 2 * x[:,:input_size]
        output = model(x, n_steps, T)

        loss = nn.MSELoss()(output[:,-1:], y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

