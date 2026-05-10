import pickle

# Path to your .pkl file
model_path = "/Users/bing/Documents/Documents - Bing’s MacBook Air/mlProfiler/tests/mps/multiinstance/output/run03112025_DL0207_0307_nonDL0311_nodvfs/seen_partition/crossvalid/throughput/trainratio_/rand10/fold_1_test_5/extratrees/14-3-2025_20:56:13_extratrees_model-corr0.0_datard10_excol.pkl"

import pickle
from sklearn.ensemble import ExtraTreesRegressor

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Verify the model type
if not isinstance(model, ExtraTreesRegressor):
    raise TypeError(f"Loaded model is not an ExtraTreesRegressor, but {type(model)}")

# Check if the model is fitted
if not hasattr(model, 'estimators_'):
    raise ValueError("Model has not been fitted. Please fit the model before inspecting.")

# Inspect each tree in the ensemble
print(f"Number of trees in the ensemble: {len(model.estimators_)}")

# Loop through each tree to get details
for i, tree in enumerate(model.estimators_):
    # Get the actual depth of the tree
    tree_depth = tree.tree_.max_depth
    # Get the number of leaves in the tree
    n_leaves = tree.tree_.n_leaves
    # Get the number of nodes in the tree
    n_nodes = tree.tree_.node_count
    # Get the feature used at the root node (as an example)
    root_feature = tree.tree_.feature[0] if tree.tree_.feature[0] != -2 else "Leaf (no split)"

    print(f"\nTree {i+1}:")
    print(f"  Actual Depth: {tree_depth}")
    print(f"  Number of Leaves: {n_leaves}")
    print(f"  Number of Nodes: {n_nodes}")
    print(f"  Feature at Root Node: {root_feature}")

# Optionally, compute the average depth across all trees
average_depth = sum(tree.tree_.max_depth for tree in model.estimators_) / len(model.estimators_)
print(f"\nAverage Depth Across All Trees: {average_depth:.2f}")

# Feature importances (if you want to see which features were most important overall)
if hasattr(model, 'feature_importances_'):
    print("\nFeature Importances:")
    for idx, importance in enumerate(model.feature_importances_):
        print(f"Feature {idx}: {importance:.4f}")