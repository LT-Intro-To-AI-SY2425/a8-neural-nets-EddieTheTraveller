from typing import List, Tuple
from neural import NeuralNet
from sklearn.model_selection import train_test_split


def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output, ensuring headers are ignored.

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list or None if header detected.
    """
    tokens = line.strip().split(",")

    # Ensure the first column is a valid number (to skip headers)
    if not tokens[0].isdigit():
        return None  # Skip headers

    out = int(tokens[0])

    # When output has 1 at the start, desired output is 0 (safe email).
    # When output has 0 at the start, desired output is 1 (phishing email).
    output = [1 if out == 1 else 0.5 if out == 2 else 0]

    inpt = [float(x) for x in tokens[1:]]  # Extract only the input features (8 total)
    return (inpt, output)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Normalizes input features to a 0-1 range.

    Args:
        data - list of (input, output) tuples

    Returns:
        Normalized data with input features mapped to 0-1 range
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


# Load and preprocess training data
with open("email_phishing_data.csv", "r") as f:
    raw_training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# Remove None values in case headers were skipped
training_data = [data for data in raw_training_data if data is not None]

train_data, test_data = train_test_split(training_data)

print("Starting data normalization...")
train_data_norm = normalize(train_data)
test_data_norm = normalize(test_data)

print("Creating NeuralNet...")
nn = NeuralNet(8, 12, 1)
nn.train(train_data_norm[:10], iters=500, print_interval=10, learning_rate=0.1)
print("Trained data")

# Manually defined test dataset (instead of reading from a CSV)
new_test_data = [
    [100, 75, 40, 0, 0, 0, 0, 0],
    [100, 68, 45, 2, 1, 1, 1, 2],
    [100, 60, 30, 4, 2, 2, 2, 4],
    [100, 80, 50, 0, 0, 0, 0, 0],
    [100, 70, 35, 3, 2, 1, 3, 3],
]

# Run predictions (corrected method)
print("\nRunning Predictions on Manual Test Data:")
for i in new_test_data:
    prediction = nn.evaluate(i)
    print(f"Input: {i}, Predicted Output: {prediction}")






