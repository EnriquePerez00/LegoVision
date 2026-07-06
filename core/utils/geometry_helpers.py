# -*- coding: utf-8 -*-
"""geometry_helpers.py
Helper functions for advanced geometry, hole subtraction, and stud detection.
"""
import cv2
import numpy as np
import math
from core.utils.config_loader import cfg

def estimate_surface_area_sam_corrected_v2(mask_cen: np.ndarray, bbox_norm: list, rest_h: float = 9.6, img_res_px: float = 1024.0, subtract_holes: bool = True) -> float:
    try:
        # Calculate raw pixels using contour hierarchy to subtract internal holes
        contours, hierarchy = cv2.findContours(mask_cen.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        total_pixels = 0.0
        if contours and hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, cnt in enumerate(contours):
                c_area = cv2.contourArea(cnt)
                parent = hierarchy[idx][3]
                if parent == -1:  # External contour
                    total_pixels += c_area
                else:  # Internal contour (hole)
                    if subtract_holes:
                        total_pixels -= c_area
        else:
            total_pixels = float(np.sum(mask_cen > 0))
            
        img_res_px_val = img_res_px
        cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
        cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
        
        cx_px = cx * img_res_px_val
        cy_px = cy * img_res_px_val
        center_px = img_res_px_val * 0.5
        
        PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
        CAMERA_DIST_MM = cfg.inference.calibration.camera_dist_mm
        
        px_per_mm_nom = PX_PER_MM_CENITAL * (img_res_px_val / 640.0)
        
        dx_mm = (cx_px - center_px) / px_per_mm_nom
        dy_mm = (center_px - cy_px) / px_per_mm_nom
        r_mm = math.sqrt(dx_mm**2 + dy_mm**2)
        
        d_cam = math.sqrt(r_mm**2 + (CAMERA_DIST_MM - rest_h)**2)
        px_per_mm = (px_per_mm_nom * CAMERA_DIST_MM) / d_cam
        
        area_raw_mm2 = total_pixels / (px_per_mm ** 2)
        
        w_bbox_mm = (bbox_norm[2] - bbox_norm[0]) * img_res_px_val / px_per_mm
        h_bbox_mm = (bbox_norm[3] - bbox_norm[1]) * img_res_px_val / px_per_mm
        perimeter_half = (w_bbox_mm + h_bbox_mm) / 2.0
        
        side_width_projected = (r_mm * rest_h) / (CAMERA_DIST_MM - rest_h)
        added_side_area_mm2 = perimeter_half * side_width_projected * 0.5
        
        area_corrected = area_raw_mm2 - added_side_area_mm2
        return max(0.1, area_corrected)
    except Exception:
        PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
        px_per_mm_nom = PX_PER_MM_CENITAL * (img_res_px / 640.0)
        return float(np.sum(mask_cen > 0)) / (px_per_mm_nom ** 2)

def detect_lego_features(mask: np.ndarray, crop_img, img_res_px: float = 1024.0) -> tuple[int, int]:
    try:
        img_np = np.array(crop_img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE to enhance local contrast of studs/acoples
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        
        # Bilateral filter to smooth texture but preserve circular borders
        blurred = cv2.bilateralFilter(gray_clahe, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 80)
        
        # Adaptive thresholding to capture low-contrast circles
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_t, hierarchy_t = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        all_contours = []
        if contours is not None:
            all_contours.extend(contours)
        if contours_t is not None:
            all_contours.extend(contours_t)
            
        if not all_contours:
            return 0, 0, 0
            
        PX_PER_MM_CENITAL = cfg.inference.calibration.px_per_mm_cenital
        px_per_mm = PX_PER_MM_CENITAL * (img_res_px / 640.0)
        r_stud_px = 2.4 * px_per_mm
        r_tube_px = 3.25 * px_per_mm
        
        min_r_stud = r_stud_px * 0.45
        max_r_stud = r_stud_px * 1.55
        min_r_tube = r_tube_px * 0.45
        max_r_tube = r_tube_px * 1.65
        
        # Check if mask shape matches crop image shape
        mask_h, mask_w = mask.shape[:2]
        crop_w, crop_h = crop_img.size
        use_mask_filter = (mask_h == crop_h and mask_w == crop_w)
        
        circular_contours = []
        for idx, cnt in enumerate(all_contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity > 0.65:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                cx, cy = int(x), int(y)
                if 0 <= cy < crop_h and 0 <= cx < crop_w:
                    if not use_mask_filter or mask[cy, cx] > 0:
                        if min_r_stud <= radius <= max_r_tube:
                            circular_contours.append({
                                'center': (x, y),
                                'radius': radius,
                            })
                            
        # Group by centers to find concentric circles
        used = set()
        groups = []
        for i, c1 in enumerate(circular_contours):
            if i in used:
                continue
            group = [c1]
            used.add(i)
            for j, c2 in enumerate(circular_contours):
                if j in used:
                    continue
                dist = np.linalg.norm(np.array(c1['center']) - np.array(c2['center']))
                if dist < 8.0:  # Concentric distance threshold
                    group.append(c2)
                    used.add(j)
            groups.append(group)
            
        studs_count = 0
        acoples_count = 0
        
        for g in groups:
            radii = [c['radius'] for c in g]
            if not radii:
                continue
            max_r = max(radii)
            min_r = min(radii)
            
            # Concentric cylinders/annuli have distinct radii (outer vs inner ring).
            # If the difference in radius is substantial (> 2.0px), it is an acople (hollow tube).
            if len(g) >= 2 and (max_r - min_r) >= 2.0:
                if max_r >= min_r_stud * 1.1:
                    acoples_count += 1
            else:
                # Same radius -> single circle (stud)
                if min_r_stud <= max_r <= max_r_stud:
                    studs_count += 1
                    
        return studs_count, acoples_count, len(groups)
    except Exception:
        return 0, 0, 0

def detect_zenith_studs(mask_cen: np.ndarray, crop_img, img_res_px: float = 1024.0) -> int:
    _, _, total_circles = detect_lego_features(mask_cen, crop_img, img_res_px)
    return total_circles

def detect_zenith_acoples(mask_cen: np.ndarray, crop_img, img_res_px: float = 1024.0) -> int:
    _, acoples, _ = detect_lego_features(mask_cen, crop_img, img_res_px)
    return acoples

def detect_frontal_studs(mask_fr: np.ndarray, crop_img=None, img_res_px: float = 1024.0) -> int:
    # 1. Circular detector (for studs facing the camera)
    circ_studs = 0
    if crop_img is not None:
        circ_studs, _, _ = detect_lego_features(mask_fr, crop_img, img_res_px)
        
    # 2. Crest profile detector (for studs on the profile)
    profile_studs = 0
    try:
        contours, _ = cv2.findContours(mask_fr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_pts = np.vstack(contours)
            rect = cv2.minAreaRect(all_pts)
            angle = rect[2]
            
            h, w = mask_fr.shape
            center = (w // 2, h // 2)
            
            rot_angle = angle
            if rect[1][0] < rect[1][1]:
                rot_angle = angle + 90
                
            M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            rotated_mask = cv2.warpAffine(mask_fr, M, (w, h), flags=cv2.INTER_NEAREST)
            
            ys, xs = np.where(rotated_mask > 0)
            if len(xs) > 0:
                x_min, x_max = np.min(xs), np.max(xs)
                top_profile = []
                for x in range(x_min, x_max + 1):
                    col_ys = ys[xs == x]
                    if len(col_ys) > 0:
                        top_profile.append(np.min(col_ys))
                    else:
                        top_profile.append(h)
                        
                top_profile = np.array(top_profile)
                height_profile = h - top_profile
                
                PX_PER_MM_FRONTAL = cfg.inference.calibration.px_per_mm_frontal
                px_per_mm = PX_PER_MM_FRONTAL * (img_res_px / 640.0)
                stud_h_px = 1.7 * px_per_mm
                stud_w_px = 4.8 * px_per_mm
                
                peaks = 0
                in_peak = False
                peak_start = 0
                baseline = np.median(height_profile)
                
                for i in range(1, len(height_profile) - 1):
                    val = height_profile[i]
                    if not in_peak:
                        if val > baseline + stud_h_px * 0.5:
                            in_peak = True
                            peak_start = i
                    else:
                        if val <= baseline + stud_h_px * 0.3:
                            in_peak = False
                            peak_w = i - peak_start
                            if stud_w_px * 0.4 <= peak_w <= stud_w_px * 1.8:
                                peaks += 1
                profile_studs = peaks
    except Exception:
        pass
        
    return max(circ_studs, profile_studs)

def detect_frontal_acoples(mask_fr: np.ndarray, crop_img=None, img_res_px: float = 1024.0) -> int:
    if crop_img is None:
        return 0
    _, acoples, _ = detect_lego_features(mask_fr, crop_img, img_res_px)
    return acoples

