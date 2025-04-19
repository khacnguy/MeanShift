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
    # Create 1D arrays for the Y and X coordinates (centered at 0)
    y = np.linspace(-(h-1)/2., (h-1)/2., h)
    x = np.linspace(-(w-1)/2.,  (w-1)/2.,  w)
    # Evaluate the 1D Gaussian on each coordinate axis
    gy = np.exp(-0.5 * (y**2) / (sigma**2))
    gx = np.exp(-0.5 * (x**2) / (sigma**2))
    # Compute the outer product to get a 2D Gaussian grid
    kernel = np.outer(gy, gx)
    return kernel

def bhattacharyya_distance(p, q):
    return np.sqrt(1 - np.sum(np.sqrt(p * q)))

def adjust_size_for_centered_pooling(x,y,original_w, original_h, window_size):
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
    Fastest 2D average pooling using NumPy reshape and sum.

    Assumes:
    - input_array shape is divisible by window_size
    - stride == window_size
    - square pooling window (window_size x window_size)
    """
    h, w = input_array.shape
    k = window_size

    # Reshape and compute average over each block
    return input_array.reshape(h // k, k, w // k, k).sum(axis=(1, 3)) / (k * k)
def prodpool2d(input_array, window_size):
    """
    2D product pooling using NumPy reshape.

    Assumes:
    - input_array shape is divisible by window_size
    - stride == window_size
    - square pooling window (window_size x window_size)
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
    
    Parameters:
    - bp: float or numpy array, backprojection value(s) in [0, 1]
    - w: float or numpy array, kernel weight(s) in [0, 1]
    - k: float, steepness of the sigmoid function (default=10)

    Returns:
    - float or numpy array in [0, 1], modulated value
    """
    sigmoid = 1 / (1 + np.exp(-k * (bp - 0.5)))
    return sigmoid * w

