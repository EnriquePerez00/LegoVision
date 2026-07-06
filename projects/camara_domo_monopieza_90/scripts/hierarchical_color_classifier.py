# -*- coding: utf-8 -*-
"""projects/camara_domo/scripts/hierarchical_color_classifier.py
================================================================
Implementa la clase HierarchicalColorClassifier que carga los modelos
jerárquicos (enrutador + especialistas) para cámara cenital y lateral,
y realiza la predicción de color (combinada o individual).
"""
import os
import json
import torch
import numpy as np

# Configurar paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

from hierarchical_router import RouterMLP
from run_evaluation import ColorMLP

class HierarchicalColorClassifier:
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Paths de modelos
        self.cenital_dir = os.path.join(project_root, "models", "hierarchical", "cenital")
        self.lateral_dir = os.path.join(project_root, "models", "hierarchical", "lateral")
        
        # Inicializar estados de carga
        self.cen_ready = False
        self.lat_ready = False
        
        # Cargar configuración cenital
        cen_meta_path = os.path.join(self.cenital_dir, "metadata.json")
        if os.path.exists(cen_meta_path):
            try:
                with open(cen_meta_path, "r", encoding="utf-8") as f:
                    self.meta_cen = json.load(f)
                
                # Cargar enrutador cenital
                self.router_cen = RouterMLP().to(self.device)
                self.router_cen.load_state_dict(torch.load(os.path.join(self.cenital_dir, "router.pt"), map_location=self.device, weights_only=True))
                self.router_cen.eval()
                
                # Cargar especialistas cenitales
                self.spec_cen = {}
                for g_id in self.meta_cen["specialists"]:
                    num_classes = len(self.meta_cen["specialists"][g_id]["classes"])
                    model = ColorMLP(num_classes=num_classes).to(self.device)
                    model.load_state_dict(torch.load(os.path.join(self.cenital_dir, f"spec{g_id}.pt"), map_location=self.device, weights_only=True))
                    model.eval()
                    self.spec_cen[int(g_id)] = model
                    
                # Scaler de enrutador cenital
                self.mean_router_cen = np.array(self.meta_cen["router"]["mean"], dtype=np.float32)
                self.scale_router_cen = np.array(self.meta_cen["router"]["scale"], dtype=np.float32)
                
                self.cen_ready = True
                print("[Hierarchical Color] Modelos cenitales cargados con éxito.")
            except Exception as e:
                print(f"[Hierarchical Color Warning] Error al cargar modelos cenitales: {e}")
                
        # Cargar configuración lateral
        lat_meta_path = os.path.join(self.lateral_dir, "metadata.json")
        if os.path.exists(lat_meta_path):
            try:
                with open(lat_meta_path, "r", encoding="utf-8") as f:
                    self.meta_lat = json.load(f)
                
                # Cargar enrutador lateral
                self.router_lat = RouterMLP().to(self.device)
                self.router_lat.load_state_dict(torch.load(os.path.join(self.lateral_dir, "router.pt"), map_location=self.device, weights_only=True))
                self.router_lat.eval()
                
                # Cargar especialistas laterales
                self.spec_lat = {}
                for g_id in self.meta_lat["specialists"]:
                    num_classes = len(self.meta_lat["specialists"][g_id]["classes"])
                    model = ColorMLP(num_classes=num_classes).to(self.device)
                    model.load_state_dict(torch.load(os.path.join(self.lateral_dir, f"spec{g_id}.pt"), map_location=self.device, weights_only=True))
                    model.eval()
                    self.spec_lat[int(g_id)] = model
                    
                # Scaler de enrutador lateral
                self.mean_router_lat = np.array(self.meta_lat["router"]["mean"], dtype=np.float32)
                self.scale_router_lat = np.array(self.meta_lat["router"]["scale"], dtype=np.float32)
                
                self.lat_ready = True
                print("[Hierarchical Color] Modelos laterales cargados con éxito.")
            except Exception as e:
                print(f"[Hierarchical Color Warning] Error al cargar modelos laterales: {e}")
                
        # Universo completo de clases (colores admisibles)
        classes_cen = []
        if self.cen_ready:
            for g_id in self.meta_cen["specialists"]:
                classes_cen.extend(self.meta_cen["specialists"][g_id]["classes"])
                
        classes_lat = []
        if self.lat_ready:
            for g_id in self.meta_lat["specialists"]:
                classes_lat.extend(self.meta_lat["specialists"][g_id]["classes"])
                
        self.all_classes = sorted(list(set(classes_cen + classes_lat)))
        
    def predict_cenital_probs(self, feat_cen):
        """Devuelve las probabilidades globales para cada clase usando solo la vista cenital."""
        if not self.cen_ready or feat_cen is None:
            return np.zeros(len(self.all_classes))
            
        # 1. Enrutar familia
        x_c_scaled = (feat_cen - self.mean_router_cen) / (self.scale_router_cen + 1e-8)
        with torch.no_grad():
            t_x_c = torch.tensor(x_c_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            p_group_cen = torch.softmax(self.router_cen(t_x_c), dim=1).cpu().numpy()[0]
            
        # 2. Ejecutar especialistas y distribuir probabilidades
        prob_cen = np.zeros(len(self.all_classes))
        for g_id, model in self.spec_cen.items():
            g_meta = self.meta_cen["specialists"][str(g_id)]
            classes_g = g_meta["classes"]
            mean_g = np.array(g_meta["mean"], dtype=np.float32)
            scale_g = np.array(g_meta["scale"], dtype=np.float32)
            
            x_g_scaled = (feat_cen - mean_g) / (scale_g + 1e-8)
            with torch.no_grad():
                t_x_g = torch.tensor(x_g_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                p_c_g = torch.softmax(model(t_x_g), dim=1).cpu().numpy()[0]
                
            for idx_local, c_name in enumerate(classes_g):
                if c_name in self.all_classes:
                    idx_global = self.all_classes.index(c_name)
                    prob_cen[idx_global] = p_group_cen[g_id] * p_c_g[idx_local]
                    
        return prob_cen

    def predict_lateral_probs(self, feat_lat):
        """Devuelve las probabilidades globales para cada clase usando solo la vista lateral."""
        if not self.lat_ready or feat_lat is None:
            return np.zeros(len(self.all_classes))
            
        # 1. Enrutar familia
        x_l_scaled = (feat_lat - self.mean_router_lat) / (self.scale_router_lat + 1e-8)
        with torch.no_grad():
            t_x_l = torch.tensor(x_l_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            p_group_lat = torch.softmax(self.router_lat(t_x_l), dim=1).cpu().numpy()[0]
            
        # 2. Ejecutar especialistas y distribuir probabilidades
        prob_lat = np.zeros(len(self.all_classes))
        for g_id, model in self.spec_lat.items():
            g_meta = self.meta_lat["specialists"][str(g_id)]
            classes_g = g_meta["classes"]
            mean_g = np.array(g_meta["mean"], dtype=np.float32)
            scale_g = np.array(g_meta["scale"], dtype=np.float32)
            
            x_g_scaled = (feat_lat - mean_g) / (scale_g + 1e-8)
            with torch.no_grad():
                t_x_g = torch.tensor(x_g_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
                p_c_g = torch.softmax(model(t_x_g), dim=1).cpu().numpy()[0]
                
            for idx_local, c_name in enumerate(classes_g):
                if c_name in self.all_classes:
                    idx_global = self.all_classes.index(c_name)
                    prob_lat[idx_global] = p_group_lat[g_id] * p_c_g[idx_local]
                    
        return prob_lat

    def predict_fused_color(self, feat_cen, feat_lat):
        """Predice el nombre del color realizando una fusión Bayesiana multiplicativa de ambas vistas."""
        p_cen = self.predict_cenital_probs(feat_cen)
        p_lat = self.predict_lateral_probs(feat_lat)
        
        # Fusión multilineal o Bayesiana
        if feat_cen is not None and feat_lat is not None:
            p_combined = p_cen * p_lat
            # Si el producto da todo ceros por falta de intersección de clases, caemos a suma
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
