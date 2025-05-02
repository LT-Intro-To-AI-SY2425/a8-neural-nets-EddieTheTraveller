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
    output = [1 if out == 0 else 0 if out == 1 else 0.5]

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
with open("new_phishing_data.csv", "r") as f:
    raw_training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# Remove None values in case headers were skipped
training_data = [data for data in raw_training_data if data is not None]

train_data, test_data = train_test_split(training_data)

print("Starting data normalization...")
train_data_norm = normalize(train_data)
test_data_norm = normalize(test_data)

print("Creating NeuralNet...")
nn = NeuralNet(8, 16, 1)
nn.train(train_data_norm[:2000], iters= 2000, print_interval=100, learning_rate=0.01)
print("Trained data")

# Manually defined test dataset (instead of reading from a CSV)
new_test_data = [
    ([160, 75, 50, 5, 3, 2, 10, 5], [1]),  # Phishing
    ([210, 68, 60, 2, 1, 1, 5, 2], [0]),  # Safe
    ([130, 60, 40, 6, 4, 3, 15, 7], [1]),  # Phishing
    ([190, 80, 55, 1, 1, 0, 3, 1], [0]),  # Safe
    ([140, 70, 45, 4, 2, 2, 12, 6], [1]),  # Phishing
    ([180, 130, 50, 3, 1, 1, 4, 2], [0]),  # Safe
]

# Run predictions with `.test_with_expected()` for comparison
print("\nRunning Predictions on Manual Test Data:")
for i in nn.test_with_expected(new_test_data):
    print(f"Expected: {i[1]}, Predicted: {i[2]}")







