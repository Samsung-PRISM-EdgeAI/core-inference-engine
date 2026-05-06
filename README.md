# Core Inference Engine - Repository Context

## Overview
This repository contains the low-level C++/Python inference kernels for Samsung's edge AI stack. It is the backbone for model execution on mobile NPUs and DSPs.

## Current Goals & Challenges
1. **Memory Optimization**: We are currently experiencing high RAM pressure on mid-tier devices (specifically Galaxy A55 series). We need to explore techniques like adaptive quantization or sparse models to reduce footprint during inference.
2. **On-Device Speed**: Inference time for vision-language models needs to drop below 150ms on mobile NPUs. Current approaches are bottlenecked by cross-attention mechanisms.
3. **Hardware-Specific Tuning**: Better utilization of specific NPU acceleration layers for quantized INT4/INT8 models.
