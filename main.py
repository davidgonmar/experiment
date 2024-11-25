import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Data setup
x_vals = np.linspace(-1.5 * np.pi, 2 / 3 * np.pi, 1000).reshape(-1, 1)
y_curve1 = np.sin(x_vals)
y_curve2 = np.sin(x_vals) + 0.7
data_for_curve1 = torch.tensor(np.hstack((x_vals, y_curve1)), dtype=torch.float32)
data_for_curve2 = torch.tensor(np.hstack((x_vals, y_curve2)), dtype=torch.float32)

# Model definition
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Tanh(),
            nn.Linear(2, 2)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(2000):
    idx1, idx2 = np.random.randint(0, len(data_for_curve1)), np.random.randint(0, len(data_for_curve2))
    x1, x2 = data_for_curve1[idx1], data_for_curve2[idx2]
    output1, output2 = model(x1), model(x2)
    loss = criterion(output1, torch.tensor([1.0, 0.0])) + criterion(output2, torch.tensor([0.0, 1.0]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.eval()


# Plot the classification of the two curves by the model
def extract_output(output):
    return "blue" if output[0] > output[1] else "red"

colors = [
    extract_output(model(data_for_curve1[i])) for i in range(len(data_for_curve1))
]
plt.scatter(x_vals, y_curve1, c=colors)
colors = [
    extract_output(model(data_for_curve2[i])) for i in range(len(data_for_curve2))
]
plt.scatter(x_vals, y_curve2, c=colors)

plt.show()


# Transformations
def map_func(x, y):
    x = torch.tensor([x, y], dtype=torch.float32)
    for i, layer in enumerate(model.layers):
        x = layer(x)
        if i == 4: # all except the last layer
            break
    assert x.shape == (2,)
    return x[0].item(), x[1].item()

def transform_data(data):
    return np.array([map_func(x, y) for x, y in data])

original_curve1 = data_for_curve1.numpy()
original_curve2 = data_for_curve2.numpy()
transformed_curve1 = transform_data(original_curve1)
transformed_curve2 = transform_data(original_curve2)

# Animation setup
def interpolate_points(start, end, t):
    return start * (1 - t) + end * t

def generate_grid(xmin, xmax, ymin, ymax, n_lines, line_points):
    return [
        [(x, y) for x in np.linspace(xmin, xmax, line_points)]
        for y in np.linspace(ymin, ymax, n_lines)
    ] + [
        [(x, y) for y in np.linspace(ymin, ymax, line_points)]
        for x in np.linspace(xmin, xmax, n_lines)
    ]

def map_grid(grid):
    return [[map_func(x, y) for x, y in line] for line in grid]

def interpolate_grid(original, transformed, t):
    return [
        [interpolate_points(np.array(o), np.array(t_), t) for o, t_ in zip(orig, trans)]
        for orig, trans in zip(original, transformed)
    ]

fig, ax = plt.subplots()
num_frames = 100
original_grid = generate_grid(-1.5 * np.pi, 2 / 3 * np.pi, -1, 1.7, 10, 100)
transformed_grid = map_grid(original_grid)

def update(frame):
    t = frame / num_frames
    interpolated_curve1 = interpolate_points(original_curve1, transformed_curve1, t)
    interpolated_curve2 = interpolate_points(original_curve2, transformed_curve2, t)
    interpolated_grid = interpolate_grid(original_grid, transformed_grid, t)

    ax.clear()
    ax.scatter(interpolated_curve1[:, 0], interpolated_curve1[:, 1], c="blue", s=1)
    ax.scatter(interpolated_curve2[:, 0], interpolated_curve2[:, 1], c="red", s=1)
    for line in interpolated_grid:
        xs, ys = zip(*line)
        ax.plot(xs, ys, color="black", linewidth=0.5, alpha=0.5)

    ax.set_title(f"Transition at t = {t:.2f}")

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
plt.show()
