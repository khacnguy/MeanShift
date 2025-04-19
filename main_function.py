import numpy as np
from utils import *

def normalize_histogram(histogram):
    histogram_sum = sum(histogram)
    return np.array([x / histogram_sum for x in histogram])

def extract_histogram(patch, nbins, weights=None):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins ** 2 + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :,
                                                                                             2]).astype(np.int32)

    # count bin indices to create histogram (use per-pixel weights if given)
    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    # zero-pad histogram (needed since bincount function does not generate histogram with nbins**3 elements)
    histogram = np.zeros((nbins ** 3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram

def backproject_histogram(patch, histogram, nbins):
    # Note: input patch must be a BGR image (3 channel numpy array)
    # convert each pixel intensity to the one of nbins bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    # calculate bin index of a 3D histogram
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins ** 2 + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :,
                                                                                             2]).astype(np.int32)

    # use histogram us a lookup table for pixel backprojection
    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection

