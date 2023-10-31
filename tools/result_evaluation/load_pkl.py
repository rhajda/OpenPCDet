import pickle

# Replace 'result.pkl' with the actual filename of your .pkl file
file_name = 'result.pkl'

try:
    with open(file_name, 'rb') as file:
        loaded_object = pickle.load(file)
        print(f"Loaded object from {file_name}: {loaded_object}")
except FileNotFoundError:
    print(f"File '{file_name}' not found.")
except Exception as e:
    print(f"An error occurred while loading the .pkl file: {e}")