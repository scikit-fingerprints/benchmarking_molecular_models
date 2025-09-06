[English](README.md) | 简体中文
# HelixDock:Pre-Training on Large-Scale Generated Docking Conformations with HelixDock to Unlock the Potential of Protein-ligand Structure Prediction Models
本仓库包含我们的[论文工作](https://arxiv.org/abs/2310.13913)代码实现。

蛋白质-配体结构预测在药物发现中至关重要，可用于确定小分子（配体）与靶蛋白（受体）之间的相互作用。传统的基于物理的对接工具虽然被广泛使用，但由于构象采样有限和评分函数不精确，其准确性受到影响。尽管一些工作尝试利用深度学习的进展提高预测准确性，但由于训练数据有限，效果仍有可以提升的空间。


HelixDock通过在传统物理对接工具生成的大规模对接构象上进行预训练，然后使用有限的经实验验证的受体-配体复合物进行微调，解决了这些挑战。这种方法显著提高了预测准确性和模型的泛化能力。经过与基于物理和深度学习的基线进行严格对比，HelixDock展示出了卓越的精度和强大的迁移能力。
HelixDock还在交叉对接和基于结构的虚拟筛选基准中表现出色，在实际的虚拟筛选项目中成功识别出高活性的抑制剂。

## 在线服务
我们同时也提供了免安装的在线预测服务[螺旋桨 HelixDock-预测](https://paddlehelix.baidu.com/app/drug/helix-dock/forecast) 。

## 许可证

本项目采用 [CC BY-NC 许可证](https://creativecommons.org/licenses/by-nc/4.0/)。

根据此许可证，您可以自由分享、复制、发布、传播作品，但需遵循以下限制：

- 署名（BY）：您必须提供适当的署名，提供指向许可证的链接，并指明是否有进行了更改。您可以使用的方式包括但不限于提供作者姓名、项目链接等信息。
- 非商业性使用（NC）：您不得将本项目用于商业目的，但可以在学术研究、教育等非商业用途下使用。

如有任何疑问，请参阅 [许可证全文](https://creativecommons.org/licenses/by-nc/4.0/legalcode)。

## 环境


## 安装
除了`requirements.txt`中列出的工具外，还需要`openbabel`工具来计算预测构象与晶体构象之间的对齐RMSD。你可以使用以下命令来安装环境。
```bash
conda create -n helixdock python=3.7
conda activate helixdock
python install -r requirements.txt
conda install openbabel==2.4.1 -c conda-forge
```
请注意，rdkit版本应为2022.3.3，否则可能在加载模型参数时会导致一些错误。

## 下载训练好的模型参数
这里我们提供了可以用来复现我们论文结果的模型参数。

```bash
mkdir -p model
wget https://paddlehelix.bd.bcebos.com/HelixDock/helixdock.pdparams
mv helixdock.pdparams ./model/
```

## 下载原始数据
```
# PDBbind core set
wget https://paddlehelix.bd.bcebos.com/HelixDock/pdbbind_core_raw.tgz
tar xzf pdbbind_core_raw.tgz
mkdir -p ../data/PDBbind_v2020/complex/
mv pdbbind_core/* ../data/PDBbind_v2020/complex/


# PoseBusters dataset
wget https://paddlehelix.bd.bcebos.com/HelixDock/posebuster_raw.tgz
tar xzf posebuster_raw.tgz
```

## 下载处理过的数据
```
mkdir -p data/processed/
# PDBbind core set
wget https://paddlehelix.bd.bcebos.com/HelixDock/pdbbind_core_processed.tgz
tar xzf pdbbind_core_processed.tgz
mv pdbbind_core_processed data/processed/

# PoseBusters dataset
wget https://paddlehelix.bd.bcebos.com/HelixDock/posebuster_processed.tgz
tar xzf posebuster_processed.tgz
mv posebuster_processed data/processed/
```


## 使用方法
为了复现我们论文的结果，我们提供了以下脚本：

```bash
#  复现PDBBind core set的结果
sh reproduce_core.sh
```

输出结果组织如下：
```
    ./log/reproduce_core/save_output/step-1
        mol_name.sdf
```

其中`mol_name.sdf`是输入分子的预测构象。


```bash
# 复现PoseBusters的结果
# 请注意，为了复现PoseBusters结果，需要多次采样并使用RTMScore和posebuster分数进行排名。
sh reproduce_posebuster.sh
```

输出结果组织如下：
```
    ./log/reproduce_posebuster/save_output/step-1
        mol_name.sdf
```

其中`mol_name.sdf`是输入分子的预测构象。

## 数据获取
为了推动小分子药物发现领域的前沿探索，为学术领域的研究者们提供最大助力，HelixDock最新技术将面向学术领域的研究人员全面开放，包括代码和亿级别的训练数据，帮助加速AI技术在小分子药物研发领域的落地，促进该领域的发展（商业客户可通过官网“合作咨询”入口咨询具体商用规则）。
  
训练数据通过如下链接联系飞桨螺旋桨PaddleHelix团队免费获取（请注明单位名称）https://paddlehelix.baidu.com/partnership

## 引用此工作

如果你在研究中使用了本仓库的代码或数据，请引用：

```bibtex
@article{liu2024pretraining,
      title={Pre-Training on Large-Scale Generated Docking Conformations with HelixDock to Unlock the Potential of Protein-ligand Structure Prediction Models}, 
      author={Lihang Liu and Shanzhuo Zhang and Donglong He and Xianbin Ye and Jingbo Zhou and Xiaonan Zhang and Yaoyao Jiang and Weiming Diao and Hang Yin and Hua Chai and Fan Wang and Jingzhou He and Liang Zheng and Yonghui Li and Xiaomin Fang},
      year={2024},
      eprint={2310.13913},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```