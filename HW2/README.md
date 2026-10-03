# Neural network classifier from scratch

A NumPy implementation of a fully connected neural network with two hidden layers. It takes two input values and predicts a class of `-1` or `+1`.

I built this while studying artificial neural networks in FFR135 / FYM135 at Chalmers. The project helped me understand matrix shapes, forward propagation, backpropagation, and stochastic gradient descent.

## How it works

The network has two inputs, eight neurons in each hidden layer, and one output neuron. Every layer uses `tanh`. Neurons subtract a threshold from their weighted input, and the sign of the final output determines the predicted class.

Training minimizes squared-error loss using one example per update. The script shuffles the training examples each epoch, trains for 350 epochs with a learning rate of `0.01`, and evaluates validation classification error after each epoch. It keeps copies of the parameters with the lowest validation error.

One completed run achieved **11.30% validation error**, below the assignment's 12% target. Initialization and shuffling are random, so a new run may produce a different result. The trained parameters from that run are not included in this repository.

## Run it

You need Python 3 and the two course datasets. They are not distributed here. Obtain them through an authorized course source and place them beside `main.py`:

```text
README.md
requirements.txt
main.py
training_set.csv
validation_set.csv
```

Each dataset must contain numeric rows with three comma-separated columns: `x1`, `x2`, and `target`. There should be no header row. Targets must be `-1` or `+1`. The original datasets have 10,000 training examples and 5,000 validation examples.

Open a terminal in this folder, then run:

```sh
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```sh
source .venv/bin/activate
```

Install the dependency and start training:

```sh
python -m pip install -r requirements.txt
python main.py
```

Training time depends on your computer. The script prints validation error after each epoch, then checks the saved best model and exports its parameters. Evaluation uses the validation set for model selection; there is no separate test-set result.

## Output files

The script writes these files beside `main.py`:

| File | Contents | Shape |
|---|---|---|
| `w1.csv` | Input-to-first-layer weights | 8 × 2 |
| `w2.csv` | First-to-second-layer weights | 8 × 8 |
| `w3.csv` | Second-layer-to-output weights | 8 × 1 |
| `t1.csv` | First-layer thresholds | 8 × 1 |
| `t2.csv` | Second-layer thresholds | 8 × 1 |
| `t3.csv` | Output threshold | 1 × 1 |

Every file uses commas as delimiters. Running the script again overwrites these files. Export happens even if the best error is above 12%, so check the printed result before using the files for submission.

## Changing the experiment

Set `M1` and `M2` to change the hidden-layer sizes. Change `epochs` to adjust training duration. The learning rate is passed as `0.01` in the training-loop call to `train_step`.

## Acknowledgments

The implementation follows the course's network equations and draws on *Machine Learning with Neural Networks* (2022) by Bernhard Mehlig. AI assistance was used for tutoring, debugging, editing, and documentation during development.

This is coursework. Confirm the course's sharing rules before making the repository public.
