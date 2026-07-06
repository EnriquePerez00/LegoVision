# -*- coding: utf-8 -*-
"""
hierarchical_router.py
======================
Defines the mapping of Lego colors to 5 physical/optical groups and 
implements the Router MLP model for enrouting.
"""

import torch
import torch.nn as nn
import numpy as np

# Define groups
GROUPS = {
    0: "transparents",
    1: "metallics",
    2: "neutrals",
    3: "warm_solids",
    4: "cold_solids"
}

# Color to Group mapping based on LEGO/BrickLink catalog names
COLOR_TO_GROUP = {
    # 0. Transparents
    "trans-dark pink": 0, "trans-red": 0, "trans-yellow": 0, "trans-green": 0,
    "trans-clear": 0, "trans-light blue": 0, "trans-brown": 0, "trans-dark blue": 0,
    "trans-neon green": 0, "trans-neon orange": 0, "trans-bright green": 0, "trans-purple": 0,
    "trans-orange": 0,
    
    # 1. Metallics & Pearls
    "pearl gold": 1, "chrome silver": 1, "metallic silver": 1, "flat silver": 1,
    "pearl titanium": 1, "chrome gold": 1, "metallic gold": 1, "metallic green": 1,
    "copper": 1,
    
    # 2. Neutrals (Low saturation / Grayscales / Earth tones)
    "black": 2, "white": 2, "dark bluish gray": 2, "light bluish gray": 2,
    "reddish brown": 2, "tan": 2, "dark tan": 2, "dark gray": 2, "light gray": 2,
    "brown": 2, "dark brown": 2, "speckle black-silver": 2, "[no color/any color]": 2,
    
    # 3. Warm Solids (Red, Yellow, Orange, Pink, Magenta)
    "red": 3, "yellow": 3, "orange": 3, "dark red": 3, "dark pink": 3,
    "bright light orange": 3, "bright light yellow": 3, "salmon": 3, "rust": 3,
    "nougat": 3, "magenta": 3, "various": 3,
    
    # 4. Cold Solids (Blue, Green, Turquoise, Lavender, Purple)
    "blue": 4, "green": 4, "medium azure": 4, "sand blue": 4, "dark turquoise": 4,
    "medium blue": 4, "dark blue": 4, "dark purple": 4, "medium lavender": 4,
    "olive green": 4, "dark green": 4, "medium violet": 4, "light green": 4
}

def get_group_id(color_name: str) -> int:
    """Returns the group index for a given color name (case-insensitive)."""
    name = color_name.strip().lower()
    
    # 0. Transparents
    if "trans-" in name or "trans " in name or "transparent" in name:
        return 0
        
    # 1. Metallics & Pearls
    if any(k in name for k in ["pearl", "chrome", "metallic", "flat", "copper", "gold", "silver", "bronze"]):
        return 1
        
    # 3. Warm Solids
    if any(k in name for k in ["red", "yellow", "orange", "pink", "magenta", "salmon", "rust", "nougat", "coral", "peach", "maroon"]):
        return 3
        
    # 4. Cold Solids
    if any(k in name for k in ["blue", "green", "azure", "lavender", "purple", "violet", "lime", "aqua", "turquoise", "teal"]):
        return 4
        
    # 2. Neutrals (Default fallback for grays, whites, blacks, browns, etc.)
    return 2

# Router MLP Model
class RouterMLP(nn.Module):
    def __init__(self, input_dim=12, num_groups=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_groups)
        )
    def forward(self, x):
        return self.net(x)
