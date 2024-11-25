import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt

x_vals = np.linspace(-math.pi * 3 / 2, math.pi * 2 / 3, 1000).reshape(-1, 1)

# Two curves not separable by a line
y_curve1 = np.sin(x_vals)
y_curve2 = np.sin(x_vals) + 0.7

data_for_curve1 = np.hstack((x_vals, y_curve1))
data_for_curve2 = np.hstack((x_vals, y_curve2))


data_for_curve1 = torch.tensor(data_for_curve1, dtype=torch.float32)
data_for_curve2 = torch.tensor(data_for_curve2, dtype=torch.float32)


plt.plot(x_vals, y_curve1)
plt.plot(x_vals, y_curve2)
plt.show()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(2, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 2)

    def forward(self, x):
        act = torch.tanh
        x = act(self.layer1(x))
        x = act(self.layer2(x))
        x = act(self.layer3(x))
        return x


model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 1000
for epoch in range(epochs):
    idx1 = np.random.randint(0, len(data_for_curve1))
    idx2 = np.random.randint(0, len(data_for_curve2))
    x1 = data_for_curve1[idx1]
    x2 = data_for_curve2[idx2]
    output1 = model(x1)
    output2 = model(x2)
    loss = criterion(output1, torch.tensor([1.0, 0.0])) + criterion(
        output2, torch.tensor([0.0, 1.0])
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Test the model
model.eval()
print(torch.softmax(model(data_for_curve1[0]), dim=0))
print(torch.softmax(model(data_for_curve2[0]), dim=0))


def extract_output(output):
    return "blue" if output[0] > output[1] else "red"


# plot the data, the colors being the output of the model
colors = [
    extract_output(model(data_for_curve1[i])) for i in range(len(data_for_curve1))
]
plt.scatter(x_vals, y_curve1, c=colors)
colors = [
    extract_output(model(data_for_curve2[i])) for i in range(len(data_for_curve2))
]
plt.scatter(x_vals, y_curve2, c=colors)

plt.show()

colormap = plt.get_cmap("viridis")


def plot_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_lines: int,
    line_points: int,
    map_func,
    plot_non_transformed: bool = False,
):
    lines_non_transformed = []
    lines_transformed = []

    for y in np.linspace(ymin, ymax, n_lines):
        lines_non_transformed.append(
            [(x, y) for x in np.linspace(xmin, xmax, line_points)]
        )
        lines_transformed.append(
            [map_func(x, y) for x in np.linspace(xmin, xmax, line_points)]
        )

    for x in np.linspace(xmin, xmax, n_lines):
        lines_non_transformed.append(
            [(x, y) for y in np.linspace(ymin, ymax, line_points)]
        )
        lines_transformed.append(
            [map_func(x, y) for y in np.linspace(ymin, ymax, line_points)]
        )

    if plot_non_transformed:
        for line in lines_non_transformed:
            xs, ys = zip(*line)
            plt.plot(xs, ys, color="purple", linewidth=0.5)

    for line in lines_transformed:
        xs, ys = zip(*line)
        plt.plot(xs, ys, color="black", linewidth=0.5)


def map_func(x, y, layer_n):
    if layer_n == 0:
        return x, y
    x = torch.tensor([x, y], dtype=torch.float32)
    output = torch.tanh(model.layer1(x))
    if layer_n == 1:
        return output[0].item(), output[1].item()

    output = torch.tanh(model.layer2(output))
    if layer_n == 2:
        return output[0].item(), output[1].item()

    output = torch.tanh(model.layer3(output))

    return output[0].item(), output[1].item()


import functools

# no layers
plot_grid(-5, 5, -5, 5, 20, 100, functools.partial(map_func, layer_n=0))

# plot also the transformed data

colors = ["blue"] * len(data_for_curve1)
plt.scatter(
    [map_func(x, y, 0)[0] for x, y in data_for_curve1],
    [map_func(x, y, 0)[1] for x, y in data_for_curve1],
    c=colors,
    s=1,
)
colors = ["red"] * len(data_for_curve2)
plt.scatter(
    [map_func(x, y, 0)[0] for x, y in data_for_curve2],
    [map_func(x, y, 0)[1] for x, y in data_for_curve2],
    c=colors,
    s=1,
)

plt.show()


# Layer 1
plot_grid(-5, 5, -5, 5, 20, 100, functools.partial(map_func, layer_n=1))

# plot also the transformed data
colors = ["blue"] * len(data_for_curve1)
plt.scatter(
    [map_func(x, y, 1)[0] for x, y in data_for_curve1],
    [map_func(x, y, 1)[1] for x, y in data_for_curve1],
    c=colors,
    s=1,
)
colors = ["red"] * len(data_for_curve2)
plt.scatter(
    [map_func(x, y, 1)[0] for x, y in data_for_curve2],
    [map_func(x, y, 1)[1] for x, y in data_for_curve2],
    c=colors,
    s=1,
)

plt.show()

# Layer 2
plot_grid(-5, 5, -5, 5, 20, 100, functools.partial(map_func, layer_n=2))

# plot also the transformed data
colors = ["blue"] * len(data_for_curve1)
plt.scatter(
    [map_func(x, y, 2)[0] for x, y in data_for_curve1],
    [map_func(x, y, 2)[1] for x, y in data_for_curve1],
    c=colors,
    s=1,
)
colors = ["red"] * len(data_for_curve2)
plt.scatter(
    [map_func(x, y, 2)[0] for x, y in data_for_curve2],
    [map_func(x, y, 2)[1] for x, y in data_for_curve2],
    c=colors,
    s=1,
)

plt.show()

# Layer 3

plot_grid(-5, 5, -5, 5, 20, 100, functools.partial(map_func, layer_n=3))

# plot also the transformed data
colors = ["blue"] * len(data_for_curve1)
plt.scatter(
    [map_func(x, y, 3)[0] for x, y in data_for_curve1],
    [map_func(x, y, 3)[1] for x, y in data_for_curve1],
    c=colors,
    s=1,
)
colors = ["red"] * len(data_for_curve2)
plt.scatter(
    [map_func(x, y, 3)[0] for x, y in data_for_curve2],
    [map_func(x, y, 3)[1] for x, y in data_for_curve2],
    c=colors,
    s=1,
)

plt.show()
