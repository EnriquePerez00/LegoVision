# -*- coding: utf-8 -*-
"""
evaluate_hierarchical_color.py
==============================
Performs 5-Fold Cross Validation for Router and Specialists.
Validates the entire End-to-End hierarchical system with late view fusion.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Setup project paths
project_root = "/Users/I764690/Code_personal/LegoVision/projects/camara_domo_75078"
sys.path.insert(0, os.path.join(project_root, "scripts"))
sys.path.insert(0, project_root)

from hierarchical_router import RouterMLP, get_group_id, COLOR_TO_GROUP
from train_and_evaluate_color_mlp import ColorMLP

CACHE_PATH = os.path.join(project_root, "data", "mlp_features_cache.npz")

def run_cross_validation(camera_name, X, y_names):
    print(f"\nEvaluating CV for camera: {camera_name.upper()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    y_groups = np.array([get_group_id(name) for name in y_names])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Router Evaluation
    router_accs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_groups)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_groups[train_idx], y_groups[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        model = RouterMLP(input_dim=12, num_groups=5).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train
        model.train()
        for epoch in range(80):
            bx = torch.tensor(X_tr_s, dtype=torch.float32).to(device)
            by = torch.tensor(y_tr, dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        with torch.no_grad():
            bval = torch.tensor(X_val_s, dtype=torch.float32).to(device)
            preds = torch.argmax(model(bval), dim=1).cpu().numpy()
            router_accs.append(np.mean(preds == y_val) * 100)
            
    print(f"-> Router Accuracy: {np.mean(router_accs):.2f}% ± {np.std(router_accs):.2f}%")
    
    # 2. Specialists Evaluation
    for g_id in range(5):
        mask = (y_groups == g_id)
        X_g = X[mask]
        y_names_g = [y_names[i] for i, m in enumerate(mask) if m]
        
        if len(X_g) < 15: # Skip if too few classes/samples
            continue
            
        unique_classes_g = sorted(list(set(y_names_g)))
        class_to_idx = {name: idx for idx, name in enumerate(unique_classes_g)}
        y_g_local = np.array([class_to_idx[name] for name in y_names_g])
        
        if len(unique_classes_g) <= 1:
            continue
            
        spec_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        spec_accs = []
        for fold, (train_idx, val_idx) in enumerate(spec_skf.split(X_g, y_g_local)):
            X_tr, X_val = X_g[train_idx], X_g[val_idx]
            y_tr, y_val = y_g_local[train_idx], y_g_local[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            
            model = ColorMLP(input_dim=12, num_classes=len(unique_classes_g)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            model.train()
            for epoch in range(100):
                bx = torch.tensor(X_tr_s, dtype=torch.float32).to(device)
                by = torch.tensor(y_tr, dtype=torch.long).to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                bval = torch.tensor(X_val_s, dtype=torch.float32).to(device)
                preds = torch.argmax(model(bval), dim=1).cpu().numpy()
                spec_accs.append(np.mean(preds == y_val) * 100)
                
        print(f"-> Specialist {g_id} ({unique_classes_g}) Accuracy: {np.mean(spec_accs):.2f}% ± {np.std(spec_accs):.2f}%")

def evaluate_end_to_end(X_cen, y_cen, X_lat, y_lat):
    print("\n==========================================")
    print("TESTING END-TO-END JERÁRQUICO CON FUSIÓN MULTIVISTA")
    print("==========================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load metadata for classes list
    meta_cen_path = os.path.join(project_root, "models", "hierarchical", "cenital", "metadata.json")
    meta_lat_path = os.path.join(project_root, "models", "hierarchical", "lateral", "metadata.json")
    
    with open(meta_cen_path, "r", encoding="utf-8") as f:
        meta_cen = json.load(f)
    with open(meta_lat_path, "r", encoding="utf-8") as f:
        meta_lat = json.load(f)
        
    # Get universe of classes
    all_classes = sorted(list(set(y_cen + y_lat)))
    
    # Load Models
    # Cenital
    router_cen = RouterMLP().to(device)
    router_cen.load_state_dict(torch.load(os.path.join(project_root, "models", "hierarchical", "cenital", "router.pt"), weights_only=True))
    router_cen.eval()
    
    # Lateral
    router_lat = RouterMLP().to(device)
    router_lat.load_state_dict(torch.load(os.path.join(project_root, "models", "hierarchical", "lateral", "router.pt"), weights_only=True))
    router_lat.eval()
    
    # Specialists Cenital
    spec_cen = {}
    for g_id in meta_cen["specialists"]:
        model = ColorMLP(num_classes=len(meta_cen["specialists"][g_id]["classes"])).to(device)
        model.load_state_dict(torch.load(os.path.join(project_root, "models", "hierarchical", "cenital", f"spec{g_id}.pt"), weights_only=True))
        model.eval()
        spec_cen[int(g_id)] = model
        
    # Specialists Lateral
    spec_lat = {}
    for g_id in meta_lat["specialists"]:
        model = ColorMLP(num_classes=len(meta_lat["specialists"][g_id]["classes"])).to(device)
        model.load_state_dict(torch.load(os.path.join(project_root, "models", "hierarchical", "lateral", f"spec{g_id}.pt"), weights_only=True))
        model.eval()
        spec_lat[int(g_id)] = model
        
    # Scalers
    scaler_router_cen = StandardScaler()
    scaler_router_cen.mean_ = np.array(meta_cen["router"]["mean"])
    scaler_router_cen.scale_ = np.array(meta_cen["router"]["scale"])
    
    scaler_router_lat = StandardScaler()
    scaler_router_lat.mean_ = np.array(meta_lat["router"]["mean"])
    scaler_router_lat.scale_ = np.array(meta_lat["router"]["scale"])
    
    # Run End to End
    correct = 0
    total = len(X_cen)
    
    for i in range(total):
        # 1. Enrutar probabilities
        # Cenital
        x_c_scaled = (X_cen[i] - scaler_router_cen.mean_) / scaler_router_cen.scale_
        with torch.no_grad():
            t_x_c = torch.tensor(x_c_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            p_group_cen = torch.softmax(router_cen(t_x_c), dim=1).cpu().numpy()[0]
            
        # Lateral
        x_l_scaled = (X_lat[i] - scaler_router_lat.mean_) / scaler_router_lat.scale_
        with torch.no_grad():
            t_x_l = torch.tensor(x_l_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            p_group_lat = torch.softmax(router_lat(t_x_l), dim=1).cpu().numpy()[0]
            
        # 2. Get specialist probabilities for all 200 colors
        prob_cen = np.zeros(len(all_classes))
        prob_lat = np.zeros(len(all_classes))
        
        # Cenital specialists runs
        for g_id, model in spec_cen.items():
            g_meta = meta_cen["specialists"][str(g_id)]
            classes_g = g_meta["classes"]
            x_g_scaled = (X_cen[i] - np.array(g_meta["mean"])) / np.array(g_meta["scale"])
            
            with torch.no_grad():
                t_x_g = torch.tensor(x_g_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                p_c_g = torch.softmax(model(t_x_g), dim=1).cpu().numpy()[0]
                
            # Distribute probabilities
            for idx_local, c_name in enumerate(classes_g):
                idx_global = all_classes.index(c_name)
                prob_cen[idx_global] = p_group_cen[g_id] * p_c_g[idx_local]
                
        # Lateral specialists runs
        for g_id, model in spec_lat.items():
            g_meta = meta_lat["specialists"][str(g_id)]
            classes_g = g_meta["classes"]
            x_g_scaled = (X_lat[i] - np.array(g_meta["mean"])) / np.array(g_meta["scale"])
            
            with torch.no_grad():
                t_x_g = torch.tensor(x_g_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                p_c_g = torch.softmax(model(t_x_g), dim=1).cpu().numpy()[0]
                
            for idx_local, c_name in enumerate(classes_g):
                idx_global = all_classes.index(c_name)
                prob_lat[idx_global] = p_group_lat[g_id] * p_c_g[idx_local]
                
        # 3. View Fusion (Bayesian Product)
        p_combined = prob_cen * prob_lat
        pred_idx = np.argmax(p_combined)
        pred_name = all_classes[pred_idx]
        
        if pred_name.lower() == y_cen[i].lower():
            correct += 1
            
    accuracy = (correct / total) * 100
    print(f"\n==========================================")
    print(f"End-to-End Hierarchical System Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"==========================================")

def main():
    if not os.path.exists(CACHE_PATH):
        print(f"[ERROR] Cache not found: {CACHE_PATH}")
        sys.exit(1)
        
    cache_data = np.load(CACHE_PATH, allow_pickle=True)
    X_cen = cache_data["X_cen"]
    y_cen = list(cache_data["y_cen"])
    X_lat = cache_data["X_lat"]
    y_lat = list(cache_data["y_lat"])
    
    # 1. Run Cross-Validation for individual parts
    run_cross_validation("cenital", X_cen, y_cen)
    run_cross_validation("lateral", X_lat, y_lat)
    
    # 2. Run final E2E test
    evaluate_end_to_end(X_cen, y_cen, X_lat, y_lat)

if __name__ == "__main__":
    main()
