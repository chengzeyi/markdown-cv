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

I published [***Comfy-WaveSpeed*** (https://github.com/chengzeyi/Comfy-WaveSpeed)](https://github.com/chengzeyi/Comfy-WaveSpeed), a SOTA inference acceleration solution for ***ComfyUI*** that has received a lot of attention from the community.
It can achieve a 2x speedup over a wide range of popular image and video generation models, including ***FLUX***, ***LTXV*** and ***HunyuanVideo***.
And according to the community feedback, it can work on a variety of hardware platforms, including NVIDIA GPUs, Apple MPS and AMD ROCm, as well as on a variety of software platforms, including Linux, Windows and MacOS.

<!--__Inference Engineer__-->
<!---->
<!--`2024.8-2024.12`-->

I published [***ParaAttention*** (https://github.com/chengzeyi/ParaAttention)](https://github.com/chengzeyi/ParaAttention), providing efficient ***Context Parallelism*** to speed up DiT image and video generation inference on multiple GPUs, as well as ***First Block Cache***, a novel technique that can speed DiT image and video generation inference by caching with minimal quality loss.
***ParaAttention*** has been adopted by leading AI inference platforms as a key component in their DiT inference acceleration pipelines.

__Software Engineer, SiliconFlow Inc__

`2023.12-2024.8`

I open-sourced the [***stable-fast*** (https://github.com/chengzeyi/stable-fast)](https://github.com/chengzeyi/stable-fast) inference performance optimization framework on ***GitHub***. This project is effective for optimizing ***HuggingFace Diffusers*** and has got over ***1000+ stars***. It is currently under active development and support for other implementations of stable diffusion is coming soon.

I also develop and maintain AIGC related inference performance optimization for SiliconFlow, including ***OneDiff*** and other projects, which achieve the best performance over all other implementations.

Now I am focusing on the development of a new acceleration framework called ***xelerate*** that utilizes the latest NVIDIA CUDA hardware features and PyTorch 2.0 APIs.
This framework can achieve peak performance on NVIDIA A100, H100 and RTX 4090 GPUs and support a wide range of optimization techniques, including INT8 and FP8 quantization,
and is expected to work seamlessly with some new models like ***FLUX***, ***CogVideoX*** and ***SD3***.
On a single NVIDIA H100 GPU, it can reduce the inference time of ***FLUX.1-dev*** for generating a single 1024x1024 image with 28 steps to 2.7 seconds,
while keeping nearly the same quality and having complete support for dynamic shape and dynamic LoRA switching.
Even on a single NVIDIA RTX 4090 GPU, it can achieve a inference speed of 6.1s per 1024x1024 image with 28 steps for ***FLUX.1-dev***.

Some key features of ***xelerate*** include:

- __A Fast Flash Attention Implementation__: This is my own implementation of ***Flash Attention*** that can achieve the best performance over all other implementations. It can outperform Dao Lab's original Flash Attention 2 implementation by at most 2x on RTX 4090 and A100.
- __Full Dynamic Shape Support__: ***xelearte*** can support dynamic shape inference out of the box, and adapt quickly to different input shapes within a few milliseconds, while TensorRT can take more than ten seconds.
- __Minimum Cold Start Time__: ***xelerate*** can compile a model and start inference within 10 seconds with the help of a efficient cache mechanism, while TensorRT can take more than 30 seconds.
- __A Complete Set of Quantization Techniques__: ***xelerate*** supports a wide range of quantization techniques, including INT8 and FP8 quantization, and is very easy to use and adjust.

__Technical Expert, Alibaba Group__

`2022-2023.10`

Primarily responsible for inference performance optimization and the development and maintenance of the Quark Intelligent Scanner project, aiming to tap into the expansive camera scanner application market. Our project employs complex deep learning models on cloud servers, distinguishing us from competitors deploying traditional computer vision algorithms locally on smartphones.

Our system mainly focuses on optimizing the TorchScript Engine, a promising deployment and optimization technique I identified two years ago. I've implemented GPU-accelerated image preprocessing, post-processing algorithms, and traditional CV image processing algorithms. After continuous enhancements emphasizing operator fusion, graph rewriting, and memory optimization, it has become stable and efficient.

I developed the ***ICE*** computational acceleration framework, which has significantly accelerated computations for Quark's online inference services. The ICE framework integrates various techniques, such as:

- __Rapid Tracing__: In PyTorch 1, model hooking achieves JIT tracing capabilities similar to torch.compile, converting code needing optimization into TorchScript format efficiently and supporting training acceleration.
- __TorchScript Graph IR Pass__: Automates computation optimizations, mainly operator fusion and replacement.
- __High-Performance Fusion Operator Library__: Implemented a series of high-performance fusion operators based on CUDNN, CUBLAS, CUDA C++, and OpenAI Triton. These operators support both forward and backward propagation, thus can also accelerate training.
- __CUDA Graph Capture, Optimized BeamSearch, Optimized Attention Layer, etc.__, further enhanced performance when combined with the aforementioned technologies.

Currently, the ICE acceleration framework has significantly accelerated models like Transformer OCR, Swin Transformer, NAFNET, Swin2SR, RealESRGAN, etc.

Recently, we aim to make a business breakthrough in the AIGC domain with the ***Stable Diffusion*** model. This project requires on-the-fly fine-tuning of the base model using personalized sample images uploaded by users, followed by extensive inference predictions. Presently, leveraging the ICE engine, I achieved an inference performance of ***60 it/s*** on NVIDIA A100 for Stable Diffusion v1.5 (512x512 resolution), delivering images in half a second, and it's compatible with ControlNet, LORA, and other...

Given our high expenditure on GPU-intensive computations, cost optimization is crucial. Our system processes over 10 million user-uploaded document images daily with fewer than 200 GPU cards, generating considerable profit. As our competitors continue to struggle with inefficient systems, their users are gradually transitioning to our product.

__Senior Software Engineer, Alibaba Group__

`2021-2022`

I was responsible for developing and maintaining our OCR and image document format restoration services. I designed an XML document format protocol for the Quark browser and, utilizing my understanding of graph algorithms from discrete mathematics, researched an algorithm to restore EXCEL table structures for our WORD/EXCEL structural restoration product. Though this system is still under active development, I transitioned roles to focus on other priorities.

Additionally, I developed a framework to integrate models into the Quark browser. This framework allows developers to write model invocation code once and deploy it across multiple platforms like cloud servers, desktops, and smartphones, each interfacing with distinct inference acceleration frameworks. This framework has facilitated over half of the client-side ML projects in the Quark browser.

__Software Engineer, Alibaba Group__

`2020-2021`

I inherited a poorly maintained GPU-based video encoding and decoding framework. Recognizing its limitations, I decided to completely rewrite the system. By studying NVIDIA CUDA programming documentation and prominent video codec frameworks like FFmpeg, I redesigned the API and data structures, rigorously testing for performance and format compatibility. This system efficiently utilizes NVCODEC for acceleration and switches to other implementations when compatibility is required. The project's rewrite pr...

Additionally, I wrote native C++ code, invoking MNN to deploy compact ML models and implemented CV algorithms in the Quark browser.

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

Open-sourced on GitHub with over ***500+ stars***.

[***Comfy-WaveSpeed*** (https://github.com/chengzeyi/Comfy-WaveSpeed)](https://github.com/chengzeyi/Comfy-WaveSpeed)

### WaveSpeedAI: Multimodal AI Acceleration Platform

__Technologies: CUDA, PyTorch, Distributed Systems, Cloud Infrastructure__

Co-founded and built WaveSpeedAI, a global platform providing unified API access to 700+ AI models with industry-leading inference speeds. The platform powers real-time image generation and video generation with up to 6x faster inference.

[***WaveSpeedAI*** (https://wavespeed.ai)](https://wavespeed.ai)

### ParaAttention (open source): Efficient Context Parallelism for DiT Inference

__Technologies: Attention Mechanism, PyTorch, Distributed Computing__

Open-sourced on GitHub with over ***100+ stars***.

[***ParaAttention*** (https://github.com/chengzeyi/ParaAttention)](https://github.com/chengzeyi/ParaAttention)

### piflux (closed source): Accelerating FLUX Inference with Multiple GPUs.

__Technologies: CUDA, PyTorch, PyTorch Distributed, Diffusion Transformer__

***piflux*** is one of the fastest FLUX inference framework with multiple GPUs.
It is not open-sourced yet. It is designed to work with ***xelerate*** seamlessly with very fine-grained sequence-level parallelism and attention KV cache strategies.
On 2 NVIDIA H100 GPUs, it can reduce the inference time of ***Flux.1-dev*** for generating a single 1024x1024 image with 28 steps to 1.7 seconds, while keeping nearly the same quality.

### xelerate (closed source): Best PyTorch Inference Optimization Framework

__Technologies: C++, CUDA, PyTorch, OpenAI Triton, TorchDynamo, TorchInductor__

***xelerate*** is the the fastest inference performance optimization framework for deep learning models.
It is not open-sourced yet. But it achieves the best performance over all other implementations.
It is on par with NVIDIA TensorRT 10, but with more flexibility and compatibility with PyTorch 2.0 APIs, and can achieve peak performance on NVIDIA A100, H100 and RTX 4090 GPUs.
It also supports a wide range of quantization techniques, including INT8 and FP8 quantization, and is expected to work seamlessly with some new models like ***FLUX*** and ***CogVideoX***.

### stable-fast (open source): A Lightweight Inference Performance Optimization Framework for Stable Diffusion

__Technologies: C++, CUDA, PyTorch, OpenAI Triton__

Open-sourced on GitHub with over ***1000+ stars***.

[***stable-fast*** (https://github.com/chengzeyi/stable-fast)](https://github.com/chengzeyi/stable-fast)

### ICE Deep Learning Computational Acceleration Framework

__Technologies: C++, CUDA, PyTorch, OpenAI Triton__

I spearheaded its development and design. This acceleration framework contains basic operator extensions, all of which include backward propagation and reduce GPU memory requirements during training through a certain level of Recompute:

- CUDNN Convolution Fusion Extension: Developed based on CUDNN V7 API, supports multiple Pattern Fusions like Conv + Bias + Addition + Activation.
- GEMM Extension: Developed based on CUBLASLT, supports multiple Pattern Fusions like GEMM + Bias + Addition + Activation, and preserves Memory Format during computation, facilitating Channels Last Propagation.
- Fused Normalization Extension: Developed using CUDA C++, supporting Norm and subsequent Pointwise computation fusion.
- Triton Op: Reimplemented PyTorch CUDA Op based on Triton, resulting in performance improvements.

This acceleration framework supports multiple frontends (TorchDynamo, TorchScript, FuncTorch) and is highly compatible with mainstream algorithm frameworks (Huggingface Transformers, Diffusers, etc.).

### NVJpeg Image Encoding Extension

__Technologies: C++, CUDA, PyTorch__

Fully compatible with PyTorch, this extension supports various sampling formats and boasts rapid speeds. On an RTX 3090Ti, it can encode over 1000 images per second.

### OCR-Based EXCEL Table Structure Restoration Algorithm

__Technologies: Python, NumPy__

This algorithm can restore intricate table structures from discrete line detection results. It's employed in multiple online services of the Quark browser, such as Quark File Scanner and Quark Table Restoration.

### Fixed-Size Memory Allocation Library Based on Multiway Trees

__Technologies: C++, Linux__

Designed with ***__builtin_ctzll***, this multiway tree data structure allows for quick memory allocation/release. In some scenarios, it's faster than TCMalloc, with minimal fragmentation, and is integrated into our stream processing framework.

### Performance Optimization of G'MIC (CImg) Image Processing Library

__Technologies: C++, Linux, OpenMP__

G'MIC is one of the most popular digital image processing frameworks among GNU users. Through OpenMP acceleration and template programming, I achieved a 4-10x performance boost in all its image processing algorithms.

<!-- ### Footer
Last updated: 2023.9 -->
