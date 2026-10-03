## HW1: Boolean functions and a perceptron

[HW1.py](HW1/HW1.py) explores which Boolean input-output mappings a single perceptron can learn.

The program:

- Generates every binary input for a chosen number of input variables.
- Trains a perceptron by updating its weights when it makes a mistake.
- Tests every possible set of output labels for two and three input variables.
- Samples 10,000 sets of output labels for four and five input variables.
- Repeats each experiment 20 times and prints the mean fraction learned and its standard deviation.

### Run HW1

Install NumPy:

```powershell
py -m pip install numpy
```

From the repository's main folder, run:

```powershell
py HW1/HW1.py
```

The experiments for four and five input variables can take a while because they train many perceptrons.

### How to interpret the results

The perceptron trains for at most 20 passes through the inputs. A failed run means it did not find a solution within those 20 passes; it does not prove that the mapping is mathematically impossible to separate with a perceptron. Random initialization and sampled label sets can also cause results to vary between runs.

