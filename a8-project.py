from typing import Tuple
from neural import *
import csv


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[0])
    #When output has 1 in the start and else has 0, then we get desired of 0(safe email). If output has 0 at the start, and else has 1, then we get desired of 1(phishing email).
    output = [1 if out == 1 else 0.5 if out == 2 else 0]

    inpt = [float(x) for x in tokens[1:]]
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("email_phishing_data.csv", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

#for line in training_data:
    #print(line)  

print("Starting data normalization...")

td = normalize(training_data)

#for line in td:
    #print(line)

print("creating NueralNet")
nn = NeuralNet(8, 12, 1)

nn.train(td[:10], iters=500, print_interval=10, learning_rate=0.1)

for i in nn.test_with_expected(td):
    print(f"desired: {i[1]}, actual: {i[2]}")

