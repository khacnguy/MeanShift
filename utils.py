import numpy as np 

def epanechnikov_kernel(size):
    h, w = size
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
    radius = np.sqrt(x**2 + y**2)
    max_radius = np.max(radius)
    weights = np.maximum(0, 1 - (radius / max_radius) ** 2).astype(np.float32)
    return weights

def gaussian_kernel(size, sigma = 6.57):
    h, w = size
    y = np.linspace(-(h-1)/2., (h-1)/2., h)
    x = np.linspace(-(w-1)/2.,  (w-1)/2.,  w)
    gy = np.exp(-0.5 * (y**2) / (sigma**2))
    gx = np.exp(-0.5 * (x**2) / (sigma**2))
    kernel = np.outer(gy, gx)
    return kernel

def bhattacharyya_distance(p, q):
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))

def adjust_size_for_centered_pooling(x,y,original_w, original_h, window_size):
    # adjust the ROI so that w, h would be a nice numbers for resizing later
    def adjust(dim):
        while dim > 0:
            if dim % window_size == 0:
                pooled = dim // window_size
                if pooled % 2 == 1:
                    return dim
            dim -= 1
        raise ValueError(f"No valid size found for dimension starting at {dim}")
    
    w_adj = adjust(original_w)
    h_adj = adjust(original_h)
    x += int((original_w - w_adj) / 2)
    y += int((original_h - h_adj) / 2)
    return x,y,w_adj, h_adj

def avgpool2d(input_array, window_size):
    """
    2D average pooling using NumPy reshape and sum.
    """
    h, w = input_array.shape
    k = window_size

    # Reshape and compute average over each block
    return input_array.reshape(h // k, k, w // k, k).sum(axis=(1, 3)) / (k * k)

def prodpool2d(input_array, window_size):
    """
    2D product pooling using NumPy reshape. (not implemented yet)
    """
    h, w = input_array.shape
    k = window_size

    # Reshape to (h // k, k, w // k, k)
    reshaped = input_array.reshape(h // k, k, w // k, k)

    # Take product over the pooling blocks
    pooled = np.prod(reshaped, axis=(1, 3))

    return pooled

def bpw(bp, w, k=8):
    """
    Modulation function combining backprojection (bp) and kernel weight (w).
    """
    sigmoid = 1 / (1 + np.exp(-k * (bp - 0.5)))
    return sigmoid * w

