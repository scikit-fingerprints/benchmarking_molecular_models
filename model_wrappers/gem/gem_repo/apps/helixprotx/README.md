English | [简体中文](README_cn.md)
# Unifying Sequences, Structures, and Descriptions for Any-to-Any Protein Generation with the Large Multimodal Model HelixProtX

Proteins are fundamental components of biological systems and can be represented through various modalities, including sequences, structures, and textual descriptions. Understanding the interrelationships between these modalities is crucial for protein research. Current deep learning approaches to protein research primarily focus on limited specialized tasks—typically predicting one type of protein modality from another. These methods limit the understanding and generation of multimodal protein data.

HelixProtX allows for the transformation of any input protein modality into any desired protein modality. 
HelixProtX consistently achieves superior accuracy across a range of protein-related tasks, outperforming existing state-of-the-art models.

## License
This project is licensed under the [CC BY-NC License](https://creativecommons.org/licenses/by-nc/4.0/).

Under this license, you are free to share, copy, distribute, and transmit the work, subject to the following restrictions:

- Attribution (BY): You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- Non-Commercial (NC): You may not use the material for commercial purposes, but you are free to use it for academic research, education, and other non-commercial purposes.

For more details, please refer to the [full license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Environment
You can use the following command to install the environment.
```bash
conda create -n helixprotx python=3.8
conda activate helixprotx
python install -r requirements.txt
```

This work does not currently open-source the model parameters of the HelixProtX modules or the LLM code. Here, we present a demo using Llama as the LLM.
Execute the following command to perform inference:
```bash
python build_model.py
python inference.py 
```

## File description
- model.py: model definition
- model_config.py: model config classes
- build_model.py: create a HelixProtX model checkpoint
- inference.py: inference code
