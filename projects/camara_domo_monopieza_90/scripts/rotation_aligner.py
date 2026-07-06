import cv2
import numpy as np
from PIL import Image

def align_image_by_moments(img: Image.Image, mask: np.ndarray = None) -> Image.Image:
    """
    Normalizes the rotation of an image by aligning its principal geometric axis vertically.
    If no mask is provided, it derives a mask from non-black pixels.
    """
    # Convert PIL Image to OpenCV format (BGR)
    img_np = np.array(img)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    else:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_np.shape[:2]

    # 1. Derive mask if not provided
    if mask is None:
        h, w = img_np.shape[:2]
        corners = np.stack([
            img_np[0, 0],
            img_np[0, w - 1],
            img_np[h - 1, 0],
            img_np[h - 1, w - 1]
        ])
        bg_color = np.median(corners, axis=0)
        diff = np.abs(img_np.astype(np.float32) - bg_color.astype(np.float32))
        mask = (np.any(diff > 15, axis=2)).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    # 2. Calculate Moments
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return img  # Return original if empty

    cX = M["m10"] / M["m00"]
    cY = M["m01"] / M["m00"]

    # Principal axis orientation angle
    mu20 = M["mu20"]
    mu02 = M["mu02"]
    mu11 = M["mu11"]
    
    # Eigenvalues of covariance matrix to calculate eccentricity
    trace = mu20 + mu02
    det = mu20 * mu02 - mu11**2
    val = (trace**2) / 4.0 - det
    val = max(0.0, val)
    lambda1 = trace / 2.0 + np.sqrt(val)
    lambda2 = trace / 2.0 - np.sqrt(val)
    
    if lambda1 > 0:
        eccentricity = np.sqrt(1.0 - (lambda2 / lambda1))
    else:
        eccentricity = 0.0

    # Skip rotation if the shape is compact/square/round (eccentricity < 0.65)
    # or if the area is very small (less than 150 pixels) to avoid noise sensitivity
    if eccentricity < 0.65 or M["m00"] < 150:
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w_box, h_box = cv2.boundingRect(coords)
            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(w, x + w_box + 2)
            y2 = min(h, y + h_box + 2)
            cropped = img_np[y1:y2, x1:x2]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        return img

    # Calculate angle of principal axis
    denom = mu20 - mu02
    if denom == 0:
        angle_rad = 0.0
    else:
        angle_rad = 0.5 * np.arctan2(2 * mu11, denom)
    
    angle_deg = np.degrees(angle_rad)

    # Rotate image and mask to align principal axis vertically (rotate by -angle_deg)
    if mu20 < mu02:
        rot_angle = -angle_deg
    else:
        rot_angle = -angle_deg + 90.0

    rot_mat = cv2.getRotationMatrix2D((cX, cY), rot_angle, 1.0)
    
    # Rotate image and mask
    aligned_img = cv2.warpAffine(img_np, rot_mat, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    aligned_mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # 4. Crop tightly around the new aligned mask
    coords = cv2.findNonZero(aligned_mask)
    if coords is not None:
        x, y, w_box, h_box = cv2.boundingRect(coords)
        # Add 2px margin
        x1 = max(0, x - 2)
        y1 = max(0, y - 2)
        x2 = min(w, x + w_box + 2)
        y2 = min(h, y + h_box + 2)
        cropped_aligned = aligned_img[y1:y2, x1:x2]
        # Return as PIL Image (RGB)
        return Image.fromarray(cv2.cvtColor(cropped_aligned, cv2.COLOR_BGR2RGB))
    
    return img
