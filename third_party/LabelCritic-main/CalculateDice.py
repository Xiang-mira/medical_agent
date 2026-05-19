import argparse
import os
import ErrorDetector as ed
import importlib

# Reload the module in case it has been updated
importlib.reload(ed)

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run SystematicComparisonLMDeploySepFigures with the specified path.')
parser.add_argument('--path', help='Path to the annotations')

# Parse the arguments
args = parser.parse_args()

# Extract the organ from the path
path = args.path
if path[-1] != '/':
    path += '/'
for organ in os.listdir(path):
    if not os.path.isdir(path + organ):
        continue

    # Call the function with the extracted organ and provided path
    ed.SaveDices(
        pth=path+organ,
        organ=organ
    )