---
layout: cv
title: 成泽毅的简历
---
# 成泽毅

__软件工程师__

<div id="webaddress">
<a href="ichengzeyi@gmail.com">ichengzeyi@gmail.com</a>
| <a href="https://www.linkedin.com/in/zeyi-cheng-37739216a/">我的LinkedIn页面</a>
| <a href="https://github.com/chengzeyi">我的GitHub页面</a>
</div>

## 教育

`2016-2020`

__学士学位：计算机科学与技术，武汉大学__

成绩/GPA：3.65/4.0

主修计算机科学，对编译器和计算机架构感兴趣。我编写了一个智能JAVA调试器作为我的毕业设计，该调试器可以启动java程序，并（理论上）通过JAVA调试协议的帮助，利用崩溃报告，自动调整程序的状态，重现崩溃，从而帮助开发人员找到崩溃的原因。


## 目前

__阿里巴巴集团技术专家(P7)__

### 专长

__深度学习推理的性能优化（CUDA，PyTorch）__

__计算机视觉算法的性能优化（CUDA，OpenCV，OpenMP）__

__Python/C++混合编程__

__GPU视频处理__

## 工作经验

__阿里巴巴集团技术专家__

`2022-现在`

主要负责推理性能优化，并开发和维护夸克智能扫描（夸克扫描王）项目，该项目旨在开发广阔的相机扫描应用市场。我们的项目在云服务器上实现复杂的深度学习模型，这使我们与那些在智能手机上本地部署传统计算机视觉算法的竞争者区别开来。

我们的系统主要侧重于TorchScript Engine的优化，这是我两年前发现的一种有前景的部署和优化技术。我还用GPU加速实现了几乎所有的图像预处理和后处理算法，以及传统CV图像处理算法。在不断改进系统，专注于算子融合、图重写和内存优化后，它已变得稳定而高效。

我开发了 ***ICE*** 计算加速框架，该框架已对夸克线上推理服务的计算取得了显著的加速效果。ICE计算加速框架主要集成了以下技术：

- __快速追踪__：在PyTorch 1中通过对模型进行Hook，实现了类似torch.compile的JIT追踪能力。快速、方便、高效地将需要优化的代码转化为TorchScript格式，并支持训练加速。
- __TorchScript Graph IR Pass__：对计算进行自动化优化，主要包括算子融合和替换。
- __高性能融合算子库__：基于CUDNN、CUBLAS、CUDA C++、OpenAI Triton等技术实现了一系列高性能融合算子，这些算子均支持正/反向传播，因此也可以用于训练加速。
- __CUDA Graph Capture、Optimized BeamSearch、Optimized Attention Layer等__：这些技术配合上述技术，对性能取得更进一步的优化成果。

目前，ICE加速框架对Transformer OCR、Swin Transformer、NAFNET、Swin2SR、RealESRGAN等模型均取得了显著的加速效果。

近期，我们希望通过 ***Stable Diffusion*** 模型，在AIGC领域取得业务突破。该项目需要用用户上传的个性化样例图片，在线上即时对基础模型进行微调训练，并进行大量的推理预测。目前，借助ICE引擎，我在NVIDIA A100上对Stable Diffusion v1.5（512x512分辨率）上取得了 ***60 it/s*** 的推理性能，实现半秒出图，并兼容ControlNet、LORA等插件。同时，对Dreambooth训练，也取得了2倍的加速效果，并降低了显存的占用。

由于我们在GPU密集计算上的高支出，成本优化至关重要。目前，我们的系统可以用不到200张GPU卡处理超过每天1000万张用户上传文档图像，产生了可观的利润。随着我们的竞争者继续在低效系统中挣扎，他们的用户正在逐渐转向我们的产品。

__阿里巴巴集团高级软件工程师__

`2021-2022`

我负责开发和维护我们的OCR和图像文档格式还原业务服务。我为夸克浏览器设计了一种XML文档格式协议，，并利用我在离散数学中对图算法的理解，为我们的WORD/EXCEL结构还原产品研究了一种EXCEL表格结构还原算法。尽管该系统目前仍在积极开发中，但我已将角色交接，以专注于其他优先事项。

此外，我开发了一个框架，将模型集成到夸克浏览器中。该框架使开发人员可以只写一次模型调用代码，然后在云服务器、桌面电脑和智能手机等多平台上部署模型，而在这些平台上分别对接了不同的推理加速框架。这个框架已经促进了夸克浏览器中超过一半的客户端ML项目。

__阿里巴巴集团软件工程师__

`2020-2021`

我继承了一个维护不善的基于GPU的视频解码和编码框架。认识到其局限性，我决定完全重写该系统。通过研究NVIDIA CUDA编程文档和著名的视频编解码框架如FFmpeg，我重新设计了API和数据结构，通过严格的测试确保性能和格式兼容性。该系统可以有效地利用NVCODEC进行加速，并在需要兼容性时切换到其他实现。该项目的重写被证明是成功的，因为它每天可以处理超过2000万个短视频。当时该工程主要支持UC出海项目VMate。

同时，我编写了本地C++代码，调用MNN来部署小型ML模型，并在夸克浏览器中实现CV算法。

__字节跳动后端软件开发实习生__

`2019-2019`

开发支付系统。

## 语言

__中文__ (母语或双语熟练)

__英语__ (职业熟练)

__雅思英语语言测试__：7.0

__GRE考试__：语文推理154，量化推理169，分析写作4.0

## 项目

### ICE深度学习计算加速框架

__技术：C++，CUDA，PyTorch，OpenAI Triton__

由我负责开发与设计，该加速框架包含以下基本算子扩展，这些算子均包含反向传播，并通过一定程度的Recompute来减少训练时的显存需求：

- CUDNN卷积融合扩展：基于CUDNN V7 API开发，支持Conv + Bias + Addition + Activation等多个Pattern的Fuse
- GEMM扩展：基于CUBLASLT开发，支持GEMM + Bias + Addition + Activation等多个Pattern的Fuse，并在计算过程中Preserve Memory Format，使得Channels Last Propagation得以完成
- Fused Normalization扩展：基于CUDA C++开发，支持Norm和后续Pointwise计算的Fuse
- Triton Op：基于Triton重新实现PyTorch CUDA Op，性能均有提升

该加速框架支持多个前端（TorchDynamo、TorchScript、FuncTorch），对主流算法框架（Huggingface Transformers、Diffusers等）均具有良好的兼容性。

### NVJpeg图像编码扩展

__技术：C++，CUDA，PyTorch__

该扩展与PyTorch 100%兼容，功能完全，支持多种采样格式，速度很快。

在RTX 3090Ti上，能够每秒编码超过1000张图像。

### 基于OCR的EXCEL表格结构恢复算法

__技术：Python，NumPy__

该算法能够从离散的线检测结果还原复杂的表格结构。

被夸克浏览器的多个在线服务（夸克文件扫描、夸克表格还原等）使用。

### 基于多叉树的固定大小内存分配库

__技术：C++，Linux__

基于***__builtin_ctzll***设计的多叉树数据结构，可快速分配/释放内存资源

在某些情况下，比TCMalloc更快，且碎片极少，集成在我们的流处理框架中。

### G'MIC（CImg）图像处理库的性能优化

__技术：C++，Linux，OpenMP__

G'MIC是GNU用户最受欢迎的数字图像处理框架之一。

通过OpenMP加速和模板编程，在其所有的图像处理算法上实现了4-10倍的性能提升。

<!-- ### Footer
Last updated: 2023年9月 -->
