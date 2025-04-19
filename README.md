# MeanShift

This repository contains an implementation of the MeanShift algorithm for object tracking, along with several enhancements and experimental evaluations. 

## File Overview

- **meanshift.py** – Original MeanShift implementation.
- **meanshift2.py** – MeanShift with window optimization.
- **meanshift3.py** – Naive window-based MeanShift.
- **meanshift4.py** – MeanShift using a different kernel weighting strategy.
- **main_function.py** – Main tracking framework integrating different versions.
- **utils.py** – Utility functions to simplify and clean up the codebase.
- **experiments.py** – Compares `meanshift`, `meanshift2`, and `meanshift3`.
- **experiments2.py** – Compares `meanshift` and `meanshift4`.
- **experiments1_result/** – Results from `experiments.py`.
- **experiments2_result/** – Results from `experiments2.py`.
- **report.pdf** – Final project report summarizing the methods and results.

## Output

- Results and output videos are saved in the `results_video/` directory (please create if not present).

## Getting Started

To run experiments (note that you have to change the hyperparameters by yourself, some experiments are already run and the results are stored in `experiments1_result` and `experiments2_result`):

```bash
unzip resized_video.zip
python3 experiments.py
python3 experiments2.py

