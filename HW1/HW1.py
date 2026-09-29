import numpy
import itertools

def test_function(inputs,answers,n):
    weights = numpy.random.normal(0,(1/n)**0.5,n)
    theta, eta = 0, 0.05

    for k in range (20):
        for i in range (len(inputs)):
            b = 0
            b = numpy.dot(weights, inputs[i]) - theta

            if b <0:
                output = -1
            else:
                output = 1


            if output == answers[i]:
                error = answers[i] - output

            else: 
                error = answers[i] - output
                weights = weights + eta * error * numpy.array(inputs[i])
                theta = theta - eta * error



    separable = True

    for i in range (len(inputs)):
        b = 0
        b = numpy.dot(weights, inputs[i]) - theta
        


        if b <0:
            output = -1
        else: 
            output = 1
        if output != answers[i]:
            separable = False
    return  (separable)

def run_experiment(n):
    inputs = list(itertools.product([0, 1], repeat=n))
    data = []

    for run in range (20):
        separable_count = 0
        for candidate_answers in itertools.product([-1, 1], repeat = len(inputs)):

            if test_function(inputs,candidate_answers,n):
                separable_count = separable_count + 1

        fraction = separable_count / 2 ** (len(inputs))
        data.append(fraction)

    average = numpy.mean(data)
    standard_deviation = numpy.std(data)
    return float(average), float(standard_deviation)

print (run_experiment(2))
print (run_experiment(3))


def run_experiment_2(n):
    inputs = list(itertools.product([0, 1], repeat=n))

    data = []

    for run in range (20):
            separable_count = 0
            for candidate_answers in range (10000):
                candidate_answers = numpy.random.choice([-1,1], size = len(inputs))

                if test_function(inputs,candidate_answers,n):
                    separable_count = separable_count + 1

            fraction = separable_count / 10000
            data.append(fraction)

    average = numpy.mean(data)
    standard_deviation = numpy.std(data)
    return float(average), float(standard_deviation)

print (run_experiment_2(4))
print (run_experiment_2(5))
