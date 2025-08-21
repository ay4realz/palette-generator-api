"""
Color utility functions for palette generation API
"""
import numpy as np
from typing import List, Tuple
import colorsys

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color format")
    
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError("Invalid hex color format")

def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """Convert RGB tuple to hex string"""
    # Ensure RGB values are in valid range
    rgb = tuple(max(0, min(255, int(val))) for val in rgb)
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def rgb_to_lab(rgb: Tuple[float, float, float]) -> List[float]:
    """Convert RGB to LAB color space"""
    # Normalize RGB values to 0-1 range
    rgb_normalized = np.array(rgb) / 255.0
    
    # Apply gamma correction
    rgb_linear = np.where(
        rgb_normalized > 0.04045,
        ((rgb_normalized + 0.055) / 1.055) ** 2.4,
        rgb_normalized / 12.92
    )
    
    # RGB to XYZ conversion matrix (sRGB to XYZ)
    conversion_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb_linear, conversion_matrix.T)
    
    # Normalize by D65 white point
    xyz_normalized = xyz / np.array([0.95047, 1.00000, 1.08883])
    
    # Apply Lab transformation
    xyz_f = np.where(
        xyz_normalized > 0.008856,
        xyz_normalized ** (1/3),
        (7.787 * xyz_normalized + 16/116)
    )
    
    l = 116 * xyz_f[1] - 16
    a = 500 * (xyz_f[0] - xyz_f[1])
    b = 200 * (xyz_f[1] - xyz_f[2])
    
    return [float(l), float(a), float(b)]

def lab_to_rgb(lab: List[float]) -> Tuple[float, float, float]:
    """Convert LAB to RGB color space"""
    l, a, b = lab
    
    # Lab to XYZ
    fy = (l + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    # Convert f values to XYZ
    xyz = np.array([
        fx**3 if fx**3 > 0.008856 else (fx - 16/116) / 7.787,
        fy**3 if fy**3 > 0.008856 else (fy - 16/116) / 7.787,
        fz**3 if fz**3 > 0.008856 else (fz - 16/116) / 7.787
    ])
    
    # Denormalize by D65 white point
    xyz = xyz * np.array([0.95047, 1.00000, 1.08883])
    
    # XYZ to RGB conversion matrix
    conversion_matrix = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb_linear = np.dot(xyz, conversion_matrix.T)
    
    # Apply gamma correction
    rgb = np.where(
        rgb_linear > 0.0031308,
        1.055 * (rgb_linear ** (1/2.4)) - 0.055,
        12.92 * rgb_linear
    )
    
    # Convert to 0-255 range and clamp
    rgb = np.clip(rgb * 255, 0, 255)
    
    return tuple(rgb.astype(float))

def generate_traditional_palette(base_color: str, palette_type: str, num_colors: int = 5) -> List[str]:
    """Generate traditional color palettes based on color theory"""
    base_rgb = hex_to_rgb(base_color)
    base_hsv = colorsys.rgb_to_hsv(base_rgb[0]/255, base_rgb[1]/255, base_rgb[2]/255)
    
    colors = [base_color]  # Start with base color
    
    if palette_type == "complementary":
        # Add complementary color (180 degrees opposite)
        comp_hue = (base_hsv[0] + 0.5) % 1.0
        comp_rgb = colorsys.hsv_to_rgb(comp_hue, base_hsv[1], base_hsv[2])
        colors.append(rgb_to_hex([c * 255 for c in comp_rgb]))
        
        # Fill remaining with variations
        for i in range(num_colors - 2):
            variation_s = max(0.1, base_hsv[1] - (i * 0.2))
            variation_v = max(0.2, base_hsv[2] - (i * 0.15))
            var_rgb = colorsys.hsv_to_rgb(base_hsv[0], variation_s, variation_v)
            colors.append(rgb_to_hex([c * 255 for c in var_rgb]))
    
    elif palette_type == "triadic":
        # Add two colors 120 degrees apart
        for offset in [1/3, 2/3]:
            tri_hue = (base_hsv[0] + offset) % 1.0
            tri_rgb = colorsys.hsv_to_rgb(tri_hue, base_hsv[1], base_hsv[2])
            colors.append(rgb_to_hex([c * 255 for c in tri_rgb]))
        
        # Fill remaining with lighter/darker versions
        for i in range(num_colors - 3):
            var_v = base_hsv[2] * (0.7 + i * 0.15)
            var_rgb = colorsys.hsv_to_rgb(base_hsv[0], base_hsv[1], min(1.0, var_v))
            colors.append(rgb_to_hex([c * 255 for c in var_rgb]))
    
    elif palette_type == "analogous":
        # Colors within 60 degrees on either side
        for offset in [-0.1, -0.05, 0.05, 0.1]:
            if len(colors) >= num_colors:
                break
            ana_hue = (base_hsv[0] + offset) % 1.0
            ana_rgb = colorsys.hsv_to_rgb(ana_hue, base_hsv[1], base_hsv[2])
            colors.append(rgb_to_hex([c * 255 for c in ana_rgb]))
    
    elif palette_type == "monochromatic":
        # Variations in saturation and value
        for i in range(1, num_colors):
            factor = i / num_colors
            var_s = max(0.1, base_hsv[1] * (1 - factor * 0.5))
            var_v = max(0.2, min(1.0, base_hsv[2] + (factor - 0.5) * 0.6))
            var_rgb = colorsys.hsv_to_rgb(base_hsv[0], var_s, var_v)
            colors.append(rgb_to_hex([c * 255 for c in var_rgb]))
    
    return colors[:num_colors]

def calculate_contrast_ratio(color1: str, color2: str) -> float:
    """Calculate WCAG contrast ratio between two colors"""
    def get_luminance(hex_color):
        rgb = [x/255.0 for x in hex_to_rgb(hex_color)]
        rgb = [x/12.92 if x <= 0.03928 else ((x+0.055)/1.055)**2.4 for x in rgb]
        return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
    
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    
    brighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return (brighter + 0.05) / (darker + 0.05)
