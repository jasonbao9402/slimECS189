# CS189-SLIM

This repository contains our CS189 project on replicating and extending SLIM-Quant for low-bit compression of large language models. We study LLaMA-2-13B under 4-bit quantization and 50% sparsity, and compare SLIM-Quant against Dense and Wanda across different calibration datasets, including C4, Pile, and CodeParrot.

## Repository contents
- `quantization.py` — quantization-related implementation
- `utils.py` — helper functions and utilities

## Project goal
Our goal is to reproduce the main empirical findings of the SLIM paper and evaluate how robust the method is under different calibration settings.

## Notes
- Main evaluation tasks include ARC-Easy, ARC-Challenge, Winogrande, and OpenBookQA.
- Some experiments require access to LLaMA-2-13B weights.
- This repository is part of a course project and is still being updated.
