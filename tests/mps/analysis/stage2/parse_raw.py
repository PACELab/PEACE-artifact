

input_path = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/analysis/stage2/1022_comb2_freqscale.raw"
# Define the structure we want to preserve
from collections import defaultdict
import sys
sys.path.append('../')

from parse_util import save_with_freqscale_individual_avg_std

def nested_defaultdict():
    return defaultdict(list)

def outer_defaultdict():
    return defaultdict(nested_defaultdict)

def parse_raw_text_to_dict(raw_text):
    """
    Parse raw text representing a defaultdict(lambda: defaultdict(list)) structure into an actual defaultdict.
    """
    # Clean up the text to replace defaultdict(<class 'list'>, ...) with actual defaultdict(lambda: defaultdict(list))
    cleaned_text = raw_text.replace(
        "defaultdict(<class 'list'>", "nested_defaultdict"
    ).replace("defaultdict(<function remove_invalid_runs", "outer_defaultdict"
    ).replace('None', 'None')  # Handle None types

    # Create an environment with our nested_defaultdict function
    env = {
        'nested_defaultdict': nested_defaultdict,
        'outer_defaultdict': outer_defaultdict,
        'None': None,  # Ensure None is recognized
        'float': float  # Ensure float is recognized if needed
    }

    # Use eval in a controlled environment
    try:
        parsed_dict = eval(cleaned_text, {"__builtins__": None}, env)
        return parsed_dict
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing text to dictionary: {e}")
        return None


with open(input_path, "r") as f:
    raw_text = f.read()
    print(raw_text)
# Convert the raw text to a dictionary
parsed_dict = parse_raw_text_to_dict(raw_text)


if parsed_dict:
    print(f"parsed_dict: {parsed_dict}")
    print("Successfully parsed dictionary!")
    #save csv
    save_with_freqscale_individual_avg_std(parsed_dict, f"1022_share_comb2_freqscale_individual_avg.csv",f"1022_share_comb2_freqscale_individual_std.csv" , 2)
    # Now you can use parsed_dict as a regular dictionary
else:
    print("Failed to parse dictionary.")
