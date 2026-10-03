# HW2: Two-layer perceptron

My second assignment for FFR135 / FYM135 Artificial Neural Networks at Chalmers.

The task was to classify two-dimensional inputs as -1 or +1 using a network with two hidden layers. I used NumPy to implement forward propagation, backpropagation, and stochastic gradient descent.

With 8 neurons in each hidden layer, a learning rate of 0.01, and 350 epochs, my best run reached 11.30% validation error. The goal was below 12%. Results can vary because the weights and training order are random.

## Running the code

Put `training_set.csv` and `validation_set.csv` in the HW2 folder beside `main.py`. The course datasets are not included.

From that folder, run:

```sh
python -m pip install -r requirements.txt
python main.py
```

The script prints validation error each epoch and saves the best weights and thresholds in six CSV files.

Based on the course material and Bernhard Mehlig's *Machine Learning with Neural Networks* (2022). I used AI assistance for learning, debugging, editing, and documentation.
