import os
import json
import torch
import torch.nn as nn
import numpy as np

class ColorMLP(nn.Module):
    def __init__(self, input_dim=12, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ColorClassifier75078_1:
    """
    Classifier specifically trained and restricted for set 75078-1.
    Loads the ColorMLP model and maps predicted indices to set 75078-1 colors.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cen_ready = True
        self.lat_ready = True
        
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(scripts_dir)
        model_path = os.path.join(project_root, "models", "color_mlp_model.pt")
        metadata_path = os.path.join(project_root, "models", "color_mlp_metadata.json")

        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.classes = self.metadata["classes"]
        self.mean_cen = np.array(self.metadata["mean_cenital"], dtype=np.float32)
        self.scale_cen = np.array(self.metadata["scale_cenital"], dtype=np.float32)
        self.mean_lat = np.array(self.metadata["mean_lateral"], dtype=np.float32)
        self.scale_lat = np.array(self.metadata["scale_lateral"], dtype=np.float32)
        
        # Load model
        self.model = ColorMLP(input_dim=12, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict_cenital_probs(self, feature_vector):
        if feature_vector is None:
            return np.zeros(len(self.classes))
        scaled = (np.array(feature_vector, dtype=np.float32) - self.mean_cen) / (self.scale_cen + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def predict_lateral_probs(self, feature_vector):
        if feature_vector is None:
            return np.zeros(len(self.classes))
        scaled = (np.array(feature_vector, dtype=np.float32) - self.mean_lat) / (self.scale_lat + 1e-8)
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def predict_fused_color(self, feat_cen, feat_lat):
        p_cen = self.predict_cenital_probs(feat_cen)
        p_lat = self.predict_lateral_probs(feat_lat)
        
        if feat_cen is not None and feat_lat is not None:
            p_combined = p_cen * p_lat
            if np.sum(p_combined) == 0:
                p_combined = p_cen + p_lat
        elif feat_cen is not None:
            p_combined = p_cen
        elif feat_lat is not None:
            p_combined = p_lat
        else:
            return "Unknown"
            
        best_idx = np.argmax(p_combined)
        return self.all_classes[best_idx]

    def predict_fused_colors_flexible(self, feat_cen, feat_lat, threshold=0.25):
        p_cen = self.predict_cenital_probs(feat_cen)
        p_lat = self.predict_lateral_probs(feat_lat)
        
        if feat_cen is not None and feat_lat is not None:
            p_combined = p_cen * p_lat
            if np.sum(p_combined) == 0:
                p_combined = p_cen + p_lat
        elif feat_cen is not None:
            p_combined = p_cen
        elif feat_lat is not None:
            p_combined = p_lat
        else:
            return ["Unknown"]
            
        sorted_indices = np.argsort(p_combined)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1]
        
        top1_prob = p_combined[top1_idx]
        top2_prob = p_combined[top2_idx]
        
        sum_prob = np.sum(p_combined)
        if sum_prob > 0:
            top1_prob /= sum_prob
            top2_prob /= sum_prob
            
        colors = [self.classes[top1_idx]]
        if (top1_prob - top2_prob) < threshold:
            colors.append(self.classes[top2_idx])
        return colors

    @property
    def all_classes(self):
        return self.classes
