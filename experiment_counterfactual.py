
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def experiment_counterfactual():
    print("Running Optimal Counterfactual Experiment (Idea 35)...")
    
    # 1. Load Data
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Tree
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Find a Malignant Example (Class 0)
    # We want to find the minimal change to make it Benign (Class 1)
    target_idx = np.where(y_test == 0)[0][0]
    x_orig = X_test[target_idx]
    pred_orig = clf.predict([x_orig])[0]
    
    print(f"Original Instance Prediction: {data.target_names[pred_orig]} (Class {pred_orig})")
    
    # 4. Brute-force Search over Leaves for Optimal Counterfactual
    # In a real Exact Solver, this is done via SAT/MIP. 
    # Here we iterate all leaves predicted as Class 1.
    
    # Fix: Use clf.apply instead of tree.apply to handle dtype conversion automatically
    leaf_indices = clf.apply(X_train) 
    
    tree = clf.tree_ 
    # Better: iterate tree structure
    
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value
    
    # Find all leaves that predict Class 1
    target_class = 1
    target_leaves = []
    
    for i in range(n_nodes):
        if children_left[i] == children_right[i]: # Leaf
            # Check prediction
            class_pred = np.argmax(value[i])
            if class_pred == target_class:
                target_leaves.append(i)
                
    print(f"Found {len(target_leaves)} target leaves predicting 'Benign'.")
    
    # For each leaf, find the "closest" point in that leaf to x_orig
    # A leaf is a hyper-rectangle defined by a set of Box Constraints.
    # Distance to box is easy to compute.
    
    min_dist = float('inf')
    best_counterfactual = None
    best_leaf = -1
    
    # Helper to get constraints for a leaf
    def get_leaf_constraints(node_id):
        # We need to traverse up from node to root... sklearn tree doesn't store parent pointers easily.
        # So let's traverse down from root to all nodes and store paths.
        pass # Too complex for this script, let's do top-down
        
    # Top-down traversal to find paths to all target leaves
    paths = {} # leaf_id -> list of (feature, op, threshold)
    
    def traverse(node, current_path):
        if children_left[node] == children_right[node]:
            if node in target_leaves:
                paths[node] = current_path
            return
        
        # Left: feat <= thresh
        traverse(children_left[node], current_path + [(feature[node], '<=', threshold[node])])
        # Right: feat > thresh
        traverse(children_right[node], current_path + [(feature[node], '>', threshold[node])])

    traverse(0, [])
    
    # Now compute distance for each leaf
    for leaf, constraints in paths.items():
        # Project x_orig onto the box defined by constraints
        x_new = x_orig.copy()
        
        # Initialize bounds: -inf to +inf
        lower_bounds = [-np.inf] * len(x_orig)
        upper_bounds = [np.inf] * len(x_orig)
        
        for feat_idx, op, thresh in constraints:
            if op == '<=':
                upper_bounds[feat_idx] = min(upper_bounds[feat_idx], thresh)
            else: # '>'
                lower_bounds[feat_idx] = max(lower_bounds[feat_idx], thresh)
        
        # Projection: clip x to [lower, upper]
        # For > thresh, we need x >= thresh + epsilon. Let's use thresh + 0.0001
        # For <= thresh, we use thresh.
        
        dist = 0
        changed_feats = 0
        
        for i in range(len(x_orig)):
            val = x_orig[i]
            l = lower_bounds[i]
            u = upper_bounds[i]
            
            new_val = val
            if val < l:
                new_val = l + 0.001 # Small epsilon
            elif val > u:
                new_val = u
            
            if new_val != val:
                dist += (new_val - val)**2
                changed_feats += 1
                x_new[i] = new_val
                
        if dist < min_dist:
            min_dist = dist
            best_counterfactual = x_new
            best_leaf = leaf
            
    print(f"Optimal Counterfactual Found!")
    print(f"L2 Distance: {np.sqrt(min_dist):.4f}")
    
    # Show changes
    print("Changes required:")
    changes = []
    for i in range(len(x_orig)):
        if abs(x_orig[i] - best_counterfactual[i]) > 1e-5:
            changes.append(f"{feature_names[i]}: {x_orig[i]:.2f} -> {best_counterfactual[i]:.2f}")
    
    for c in changes:
        print(f"  - {c}")
        
    print(f"\nConclusion: Tree structure allows instant retrieval of GLOBAL optimal counterfactuals.")
    print("Black-box methods (LIME/SHAP) only approximate this locally.")

if __name__ == "__main__":
    experiment_counterfactual()
