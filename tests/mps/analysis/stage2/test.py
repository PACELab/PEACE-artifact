import pickle
import sys
input_path = sys.argv[1]
with open(input_path, 'rb') as f:
    data = pickle.load(f)

print(data)