---
layout: cv
title: Cheng Zeyi's CV
---
# Cheng Zeyi (成泽毅)

__Founder & CEO, WaveSpeedAI__

<div id="webaddress">
<a href="ichengzeyi@gmail.com">ichengzeyi@gmail.com</a>
| Tel: +86 156 2907 7560
| <a href="https://www.linkedin.com/in/zeyi-cheng-37739216a/">My LinkedIn Page</a>
| <a href="https://github.com/chengzeyi">My GitHub Page</a>
</div>

## Education

`2016-2020`

__Bachelor of Science in Computer Science and Technology, Wuhan University__

GPA: 3.65/4.0

Majored in Computer Science with a keen interest in compilers and computer architecture. For my graduation project, I developed a smart JAVA debugger that can launch a JAVA program and, theoretically, with the aid of the JAVA Debug Protocol, adjust the program's state using crash reports to reproduce crashes, assisting developers in identifying the cause.

## Current Role

__Founder & CEO of WaveSpeedAI — Building the Fastest Infrastructure for AI Image and Video Generation__

### Expertise

__Performance Optimization for Deep Learning Inference (CUDA, CUTLASS, Triton, TorchInductor)__

__Performance Optimization for Computer Vision Algorithms (CUDA, OpenCV, OpenMP)__

__Python/C++ Hybrid Programming__

__GPU Video Processing__

## Work Experience

__Founder & CEO, WaveSpeedAI__

`2025.3-Present`

Co-founded [***WaveSpeedAI***](https://wavespeed.ai), a Singapore-based company building the core acceleration engine for the multimodal AI era.

WaveSpeedAI aggregates 700+ advanced AI models for image, video, and audio generation, making them faster, more efficient, and accessible. The platform delivers up to 6x faster inference and reduces compute costs by up to 67% compared to traditional cloud solutions.

Key achievements:
- Became an official ***Hugging Face Inference Provider***, enabling developers worldwide to use WaveSpeedAI as their acceleration engine
- Gained trust of developers and companies globally within months of launch
- Platform powers real-time FLUX image generation and video generation with industry-leading speeds

__Inference Engineer__

`2024.8-2025.3`

<!--__T11 Engineer, Tencent__-->
<!---->
<!--`2024.12-Present`-->

Published [***Comfy-WaveSpeed***](https://github.com/chengzeyi/Comfy-WaveSpeed), a state-of-the-art inference acceleration solution for ***ComfyUI*** achieving 2x speedup on popular image and video generation models including ***FLUX***, ***LTXV***, and ***HunyuanVideo***. Cross-platform support for NVIDIA GPUs, Apple MPS, AMD ROCm on Linux, Windows, and macOS.

<!--__Inference Engineer__-->
<!---->
<!--`2024.8-2024.12`-->

Published [***ParaAttention***](https://github.com/chengzeyi/ParaAttention), providing efficient ***Context Parallelism*** for multi-GPU DiT inference acceleration, along with ***First Block Cache***, a novel caching technique that accelerates DiT inference with minimal quality loss. Adopted by leading AI inference platforms as a core component of their acceleration pipelines.

__Software Engineer, SiliconFlow Inc__

`2023.12-2024.8`

Open-sourced [***stable-fast***](https://github.com/chengzeyi/stable-fast), an inference optimization framework for ***HuggingFace Diffusers*** with ***1000+ stars*** on GitHub.

Developed and maintained AIGC inference optimizations for SiliconFlow, including ***OneDiff***, achieving best-in-class performance.

Built ***xelerate***, an acceleration framework leveraging NVIDIA CUDA hardware features and PyTorch 2.0 APIs. Achieved peak performance on A100, H100, and RTX 4090 GPUs with INT8/FP8 quantization support. Key benchmarks:
- ***FLUX.1-dev*** on H100: 2.7s per 1024x1024 image (28 steps)
- ***FLUX.1-dev*** on RTX 4090: 6.1s per 1024x1024 image (28 steps)
- Full support for dynamic shapes and dynamic LoRA switching

Key features of ***xelerate***:
- __Custom Flash Attention__: Up to 2x faster than Dao Lab's Flash Attention 2 on RTX 4090 and A100
- __Dynamic Shape Support__: Adapts to different input shapes in milliseconds (vs. 10+ seconds for TensorRT)
- __Fast Cold Start__: Model compilation and inference start within 10 seconds (vs. 30+ seconds for TensorRT)
- __Quantization__: Comprehensive INT8 and FP8 quantization support

__Technical Expert, Alibaba Group__

`2022-2023.10`

Led inference performance optimization for Quark Intelligent Scanner, a cloud-based document scanning service processing ***10+ million images daily*** with fewer than 200 GPUs.

Built GPU-accelerated image preprocessing pipelines and optimized TorchScript Engine with operator fusion, graph rewriting, and memory optimization.

Developed ***ICE*** acceleration framework for Quark's inference services:
- __Rapid Tracing__: torch.compile-like JIT tracing for PyTorch 1 via model hooking
- __TorchScript Graph IR Pass__: Automated operator fusion and replacement
- __Fusion Operator Library__: High-performance operators using CUDNN, CUBLAS, CUDA C++, and Triton
- __CUDA Graph Capture__: Optimized BeamSearch and Attention layers

Accelerated Transformer OCR, Swin Transformer, NAFNET, Swin2SR, and RealESRGAN models.

Led AIGC initiatives with ***Stable Diffusion***, enabling on-the-fly fine-tuning with user-uploaded images. Achieved ***60 it/s*** inference on NVIDIA A100 for Stable Diffusion v1.5 (512x512), delivering images in under a second with full ControlNet and LoRA compatibility.

__Senior Software Engineer, Alibaba Group__

`2021-2022`

Developed OCR and document format restoration services for Quark browser. Designed XML document protocol and implemented graph-based algorithm for EXCEL table structure restoration.

Built cross-platform ML model deployment framework supporting cloud, desktop, and mobile with unified APIs. Framework powers over half of Quark's client-side ML projects.

__Software Engineer, Alibaba Group__

`2020-2021`

Rewrote a GPU-based video encoding/decoding framework from scratch. Redesigned APIs and data structures based on NVIDIA CUDA documentation and FFmpeg patterns, with rigorous performance and compatibility testing. The system uses NVCODEC for acceleration with fallback for compatibility, processing over 20 million short videos daily.

Deployed compact ML models via MNN and implemented CV algorithms in native C++ for Quark browser.

__Backend Software Development Intern, ByteDance__

`2019`

Developed payment systems.

## Languages

__Chinese__ (Native or Bilingual Proficiency)

__English__ (Professional Proficiency)

__IELTS English Language Test__: 7.0

__GRE Exam__: Verbal Reasoning 154, Quantitative Reasoning 169, Analytical Writing 4.0

## Projects

### Comfy-WaveSpeed (open source): A SOTA Inference Acceleration Solution for ComfyUI

__Technologies: Dynamic Caching, PyTorch, ComfyUI__

***500+ stars*** on GitHub. [View Project](https://github.com/chengzeyi/Comfy-WaveSpeed)

### WaveSpeedAI: Multimodal AI Acceleration Platform

__Technologies: CUDA, PyTorch, Distributed Systems, Cloud Infrastructure__

Co-founded and built WaveSpeedAI, a global platform providing unified API access to 700+ AI models with industry-leading inference speeds. The platform powers real-time image generation and video generation with up to 6x faster inference.

[View Platform](https://wavespeed.ai)

### ParaAttention (open source): Efficient Context Parallelism for DiT Inference

__Technologies: Attention Mechanism, PyTorch, Distributed Computing__

***100+ stars*** on GitHub. [View Project](https://github.com/chengzeyi/ParaAttention)

### piflux (closed source): Accelerating FLUX Inference with Multiple GPUs.

__Technologies: CUDA, PyTorch, PyTorch Distributed, Diffusion Transformer__

Multi-GPU FLUX inference framework with fine-grained sequence-level parallelism and attention KV cache strategies. Integrates seamlessly with ***xelerate***. Achieves ***1.7s*** per 1024x1024 image (28 steps) on 2x H100 GPUs with near-original quality.

### xelerate (closed source): Best PyTorch Inference Optimization Framework

__Technologies: C++, CUDA, PyTorch, OpenAI Triton, TorchDynamo, TorchInductor__

High-performance inference optimization framework matching TensorRT 10 performance with superior PyTorch 2.0 compatibility. Achieves peak performance on A100, H100, and RTX 4090 GPUs with INT8/FP8 quantization. Powers ***FLUX*** and ***CogVideoX*** inference.

### stable-fast (open source): A Lightweight Inference Performance Optimization Framework for Stable Diffusion

__Technologies: C++, CUDA, PyTorch, OpenAI Triton__

***1000+ stars*** on GitHub. [View Project](https://github.com/chengzeyi/stable-fast)

### ICE Deep Learning Computational Acceleration Framework

__Technologies: C++, CUDA, PyTorch, OpenAI Triton__

Internal acceleration framework with operator extensions supporting both forward and backward propagation:
- CUDNN Convolution Fusion (Conv + Bias + Addition + Activation)
- CUBLASLT GEMM Fusion with Channels Last propagation
- Fused Normalization with Pointwise fusion
- Triton-based PyTorch CUDA Op reimplementations

Compatible with TorchDynamo, TorchScript, FuncTorch, and mainstream frameworks (Transformers, Diffusers).

### NVJpeg Image Encoding Extension

__Technologies: C++, CUDA, PyTorch__

PyTorch-compatible GPU image encoding extension. Encodes 1000+ images per second on RTX 3090Ti with support for various sampling formats.

### OCR-Based EXCEL Table Structure Restoration Algorithm

__Technologies: Python, NumPy__

Restores complex table structures from discrete line detection results. Powers Quark File Scanner and Quark Table Restoration services.

### Fixed-Size Memory Allocation Library Based on Multiway Trees

__Technologies: C++, Linux__

High-performance memory allocator using multiway tree data structures with `__builtin_ctzll`. Outperforms TCMalloc in certain scenarios with minimal fragmentation. Integrated into production stream processing framework.

### Performance Optimization of G'MIC (CImg) Image Processing Library

__Technologies: C++, Linux, OpenMP__

Achieved 4-10x performance improvement across all image processing algorithms in G'MIC, a popular GNU image processing framework, using OpenMP acceleration and template programming.

<!-- ### Footer
Last updated: 2023.9 -->
