[English](README.md) | 简体中文
# Unifying Sequences, Structures, and Descriptions for Any-to-Any Protein Generation with the Large Multimodal Model HelixProtX


蛋白质是生物系统的基本组成部分，可以通过各种模态来表示，包括序列、结构和文本描述。理解这些模态之间的相互关系对蛋白质研究至关重要。当前对蛋白质研究的深度学习方法主要侧重于有限的专门任务——通常从一种蛋白质模态预测另一种蛋白质模态。这些方法限制了对多模态蛋白质数据的理解和生成。

HelixProtX允许将任何输入蛋白质模态转化为任何所需的蛋白质模态。
HelixProtX在一系列与蛋白质相关的任务中始终保持卓越的准确性，优于现有的sota模型。




## 许可证

本项目采用 [CC BY-NC 许可证](https://creativecommons.org/licenses/by-nc/4.0/)。

根据此许可证，您可以自由分享、复制、发布、传播作品，但需遵循以下限制：

- 署名（BY）：您必须提供适当的署名，提供指向许可证的链接，并指明是否有进行了更改。您可以使用的方式包括但不限于提供作者姓名、项目链接等信息。
- 非商业性使用（NC）：您不得将本项目用于商业目的，但可以在学术研究、教育等非商业用途下使用。

如有任何疑问，请参阅 [许可证全文](https://creativecommons.org/licenses/by-nc/4.0/legalcode)。



## 安装
你可以使用以下命令来安装环境。
```bash
conda create -n helixprotx python=3.8
conda activate helixprotx
python install -r requirements.txt
```


## 推理流程
本工作暂不开源HelixProtX各个模块的模型参数以及LLM模型代码，这里展示一个以Llama作为LLM的demo。

执行以下命令做inference：
```bash
python build_model.py
python inference.py 
```

## 文件说明
- model.py: 模型定义
- model_config.py: model config classes
- build_model.py: 创建初始化checkpoint，用于展示
- inference.py: 推理代码

