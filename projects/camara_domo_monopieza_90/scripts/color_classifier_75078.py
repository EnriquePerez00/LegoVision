import os
import json
import torch
import torch.nn as nn
import numpy as np

class ColorCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 -> 16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class ColorClassifier75078:
    """
    Classifier specifically trained and restricted for set 75078-1.
    Loads the ColorCNN model and maps predicted indices to set 75078-1 colors.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cen_ready = True
        self.lat_ready = True
        
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(scripts_dir)
        model_path = os.path.join(project_root, "models", "color_mlp_model_75078.pt")
        metadata_path = os.path.join(project_root, "models", "color_mlp_metadata_75078.json")

        
        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.classes = self.metadata["classes"]
        
        # Load model
        self.model = ColorCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict_cenital_probs(self, crop_tensor):
        if crop_tensor is None:
            return np.zeros(len(self.classes))
            
        x_tensor = torch.tensor(crop_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def predict_lateral_probs(self, crop_tensor):
        return np.zeros(len(self.classes))

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
