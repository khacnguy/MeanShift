import numpy as np
from utils import *

def normalize_histogram(histogram):
    histogram_sum = sum(histogram)
    return np.array([x / histogram_sum for x in histogram])

def extract_histogram(patch, nbins, weights=None):
    # get the number of pixels in each bins
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins ** 2 + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :,
                                                                                             2]).astype(np.int32)

    if weights is not None:
        histogram_ = np.bincount(bin_idxs.flatten(), weights=weights.flatten())
    else:
        histogram_ = np.bincount(bin_idxs.flatten())
    histogram = np.zeros((nbins ** 3, 1), dtype=histogram_.dtype).flatten()
    histogram[:histogram_.size] = histogram_
    return histogram

def backproject_histogram(patch, histogram, nbins):
    # get the weight of each pixels
    channel_bin_idxs = np.floor((patch.astype(np.float32) / float(255)) * float(nbins - 1))
    bin_idxs = (channel_bin_idxs[:, :, 0] * nbins ** 2 + channel_bin_idxs[:, :, 1] * nbins + channel_bin_idxs[:, :,
                                                                                             2]).astype(np.int32)

    backprojection = np.reshape(histogram[bin_idxs.flatten()], (patch.shape[0], patch.shape[1]))
    return backprojection

