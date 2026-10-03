# Developed with AI-assisted tutoring, debugging, and editing.
#Implemented using Algorithm 4 on p.101 in "Machine Learning with Neural Networks" 2022 by Bernhard Mehlig.

from pathlib import Path
import numpy as np

# Find data and save results beside this script.
project_folder = Path(__file__).resolve().parent

training_data = np.loadtxt(
    project_folder / "training_set.csv",
    delimiter=","
)

validation_data = np.loadtxt(
    project_folder / "validation_set.csv",
    delimiter=","
)

M1 = 8
M2 = 8

#Input to layer 1
w1 = np.random.normal(loc = 0, scale = 0.3, size =(M1, 2))
#Layer 1 to layer 2
w2 = np.random.normal(loc = 0, scale = 0.3, size =(M2, M1))
#Layer 2 to output
w3 = np.random.normal(loc = 0, scale = 0.3, size =(M2, 1))

t1 = np.zeros(shape = M1)
t2 = np.zeros(shape = M2)
t3 = 0.0

X_train = training_data[:, :2]
y_train = training_data[:, 2]

X_val = validation_data[:, :2]
y_val = validation_data[:, 2]

def forward (x, w1, w2, w3, t1, t2, t3):


    z1 = (w1 @ x) - t1
    v1 = np.tanh(z1)

    z2 = (w2 @ v1) - t2
    v2 = np.tanh(z2)

    z3 = (w3.T @ v2) - t3
    output = np.tanh(z3)

    return v1, v2, output

# Start with the output error and work backwards through the layers using the chain rule. Every weight gradient uses its layer's error
# and its input.
# Threshold Gradients have the opposite sign because thresholds are subtracted.
def backward (x, target, v1, v2, output, w2, w3):
    delta3 = (output - target) * (1 - output ** 2)

    v2_column = np.reshape(v2, (v2.size,1))

    grad_w3 = v2_column * delta3

    w3_vector = np.reshape(w3, (v2.size,))
    delta2 = delta3 * w3_vector * (1 - v2 **2)

    grad_w2 = np.outer(delta2,v1)

    delta1 = (w2.T @ delta2) * (1 - v1 ** 2)


    grad_w1 = np.outer(delta1, x)
    grad_t1 = -delta1
    grad_t2 = -delta2
    grad_t3 = -delta3

    return grad_w1, grad_w2, grad_w3, grad_t1, grad_t2, grad_t3

# Train one example first, run the forward pass, calculate its gradients, then subtract the learning rate times each gradient from the parameters.
def train_step (x, target, learning_rate, w1, w2, w3, t1, t2, t3):

    v1, v2, output = forward(x, w1, w2, w3, t1, t2, t3)

    grad_w1, grad_w2, grad_w3, grad_t1, grad_t2, grad_t3 = backward(x, target, v1, v2, output, w2, w3)

    w1 = w1 - learning_rate * (grad_w1)
    w2 = w2 - learning_rate * (grad_w2)
    w3 = w3 - learning_rate * (grad_w3)

    t1 = t1 - learning_rate * (grad_t1)
    t2 = t2 - learning_rate * (grad_t2)
    t3 = t3 - learning_rate * (grad_t3)

    return w1, w2, w3, t1, t2, t3

def validation_error(X_val, y_val, w1, w2, w3, t1 ,t2 ,t3):
    total_error = 0
    for i in range (len(X_val)):
        output = forward(X_val[i], w1, w2, w3, t1, t2, t3)[2]
        error_contribution =np.abs(np.sign(output) - y_val[i])
        total_error += error_contribution

    C = total_error / (2 * len(X_val))
    return C

epochs = 350
error = validation_error(X_val, y_val, w1, w2, w3, t1, t2, t3)

best_error = error.item()

best_w1 = w1.copy()
best_w2 = w2.copy()
best_w3 = w3.copy()

best_t1 = t1.copy()
best_t2 = t2.copy()
best_t3 = np.array(t3, copy=True)

# A new random order each epoch keeps SGD from following the same sequence
# of examples every time. Copy the parameters when validation improves so
# later updates cannot overwrite the best version found.

for j in range(epochs):
    indices = np.random.permutation(len(X_train))

    for i in indices:
        x = X_train[i]
        target = y_train[i]

        w1, w2, w3, t1, t2, t3 = train_step(x, target, 0.01, w1, w2, w3, t1, t2, t3)

    error = validation_error(X_val, y_val, w1, w2, w3, t1, t2, t3)
    print(f"Epoch {j + 1}: validation error {error.item() * 100:.2f}%")

    if error.item() < best_error:
        best_error = error.item()
        best_w1 = w1.copy()
        best_w2 = w2.copy()
        best_w3 = w3.copy()

        best_t1 = t1.copy()
        best_t2 = t2.copy()
        best_t3 = t3.copy()

verified_error = validation_error(
    X_val, y_val,
    best_w1, best_w2, best_w3,
    best_t1, best_t2, best_t3
).item()

print(f"Best recorded error: {best_error * 100:.2f}%")
print(f"Verified error: {verified_error * 100:.2f}%")

folder = project_folder

# Save the best weights and thresholds in the six required CSV files.
# Reshape the threshold vectors as columns and the scalar t3 as a 1×1 array
# so each file has a consistent two-dimensional CSV layout.
np.savetxt(folder / "w1.csv", best_w1, delimiter=",")
np.savetxt(folder / "w2.csv", best_w2, delimiter=",")
np.savetxt(folder / "w3.csv", best_w3.reshape(-1, 1), delimiter=",")

np.savetxt(folder / "t1.csv", best_t1.reshape(-1, 1), delimiter=",")
np.savetxt(folder / "t2.csv", best_t2.reshape(-1, 1), delimiter=",")
np.savetxt(
    folder / "t3.csv",
    np.asarray(best_t3).reshape(1, 1),
    delimiter=","
)

print("Saved all six parameter files in:", folder)
