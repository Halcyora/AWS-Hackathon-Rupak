import numpy as np
import cv2

def calculate_fractal_dimension(image_array):
    """Calculates the global fractal dimension using box-counting."""
    # Binarize the image using Otsu's thresholding to find structures
    _, binary = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pixels = (binary > 0)
    
    # Box sizes (powers of 2)
    min_dim = min(pixels.shape)
    n = int(np.floor(np.log2(min_dim)))
    sizes = 2 ** np.arange(n, 1, -1)
    
    counts = []
    for size in sizes:
        # Reshape to boxes and count if any pixel exists in the box
        shape = (pixels.shape[0] // size, size, pixels.shape[1] // size, size)
        reshaped = pixels[:shape[0]*size, :shape[2]*size].reshape(shape)
        count = reshaped.any(axis=(1, 3)).sum()
        counts.append(count)
        
    # Fit line to log-log plot to find the dimension (slope)
    if len(counts) > 1:
        coeffs = np.polyfit(np.log(1/sizes), np.log(counts), 1)
        return float(coeffs[0])
    return 0.0

def generate_entropy_heatmap(image_array, grid_size=8):
    """Breaks image into a grid, calculates FD for each, and maps to colors."""
    h, w = image_array.shape
    h_step, w_step = h // grid_size, w // grid_size
    
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for i in range(grid_size):
        for j in range(grid_size):
            block = image_array[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
            fd = calculate_fractal_dimension(block)
            heatmap[i, j] = fd
            
    # Normalize heatmap to 0-255 for visualization
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Resize back to original image size
    heatmap_resized = cv2.resize(heatmap_normalized, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Apply JET colormap (Blue = Low Entropy/Normal, Red = High Entropy/Anomaly)
    color_map = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Simulate a threshold: only show high entropy areas (mocking the AI inference)
    mean_val = np.mean(heatmap_normalized)
    mask = heatmap_resized > (mean_val * 1.2) # Threshold multiplier
    
    final_overlay = np.zeros_like(color_map)
    final_overlay[mask] = color_map[mask]
    
    return final_overlay, heatmap