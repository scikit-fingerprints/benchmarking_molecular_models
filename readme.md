# Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning

[![arXiv](https://img.shields.io/badge/arXiv-2508.06199-b31b1b.svg)](https://arxiv.org/abs/2508.06199)

---

This is the repository containing the code for the paper "Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning" by Mateusz Praski, Jakub Adamczyk, Wojciech Czech. [link](https://arxiv.org/abs/2508.06199)

## Requirements

- Python >= 3.10, < 3.12
- Pyenv for managing embedding model venvs

## Installation

After you create a fresh venv (Python >=3.10,<3.12), run the following command to install the dependencies:

```sh
./install_deps.sh
```

## Usage
Start with downloading all of the required datasets. By default, all TDC ADMET + MoleculeNet benchmarks can be downloaded using the command:
```sh
python download.py
```

To run the embedding procedure for an existing model, for all datasets, run the command:
```sh
./embed_wrapper.sh /path/to/implementation # e.g. ./embed_wrapper.sh model_wrappers/huggingface
```

If you want to run this in the background, you can use the shorthand command:
```sh
./run_embed.sh <implementation name> # e.g. ./run_embed.sh huggingface
```

This will perform the embedding procedure on all defined datasets (see `config/embed.yaml` for the default list).

To train and evaluate supervised learning heads using the embeddings, run the following command:
```sh
python score.py --multirun +experiment=<implementation name> # e.g +experiment=huggingface
```

If you want to run this in the background, you can use the shorthand command:
```sh
./run_scoring.sh <implementation name> # e.g. ./run_scoring.sh huggingface
```

List of available implementations is defined under `config/experiment`.
List of available datasets is defined under `config/dataset`.

Dataset named `ogbg-molXXX` refers to the `XXX` dataset from the [MoleculeNet](https://moleculenet.org/). The splits are taken from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/).

Embeddings will be saved as a [EmbeddedDataset](src/common/types.py) object under `data/embedded/<dataset>/model.<joblib/json>`, depending on the implementation. Due to the Pandas serialization format incompatibility in different environments, we implement a simple custom serialization of outputs from some older models.

SQLite database with the best-performing classifier heads (based on the grid search tuning) will be saved to `data/meta.db`.

In our benchmark study, some of the models required filtering of a small number of SMILES, e.g.: due to the SELFIES translation issues, or lack of support for dative bonds. Those SMILES are defined in [illegal_smiles.txt](config/illegal_smiles.txt) in the config directory. To reproduce those illegal SMILES as in the original articles, please refer to the `selfies_debugger.ipynb` notebook.

To generate illustrations referred in the article, please refer to the `visualizations.ipynb`.

To generate BBT models, please follow the [BBTcomp installation setup](https://github.com/jwainer/bbtcomp) with PyStan installation. After that step, generate the `classificationreport.csv` using `visualizations.ipynb`, and run the `bbt.ipynb` to generate the BBT model statistics.

## Testing your own model

To test your own model, please implement a wrapper following the [instructions here](docs/custom_model.md). You can also refer to the existing implementations under `model_wrappers` directory.

If you want to only test your own model, we recommend using the `lightweight` branch, which only contains the benchmarking procedure source code, without any models. You can use our results as a reference point, just copy-and-paste `results/arxiv_preprint_2025_08.db` file to `data/meta.db` to append your results to the existing ones.

### Sharing your model

If you want your model to be included in our latest benchmark, feel free to open a PR with your implementation. We will be happy to review it and include it in the results.

## Tips on controlling number of threads

This project relies on hydra multirun functionality to run experiments in parallel. By default, the embedding procedure does not perform multirun, however if you want to run multiple datasets at once, you can specify parameter `hydra.launcher.n_jobs` in the experiment config (`config/experiment/model_name.yaml`, see [fingerprints](config/experiment/fingerprints.yaml) for an example). When using multirun mode, you also need to override hydra launcher from default to `joblib` (as in fingerprints example). See more about hydra multirun [here](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) and [here](https://hydra.cc/docs/plugins/joblib_launcher/).

Scoring procedure can also be run in multirun mode, in a similar manner. The default configuration provides 8 datasets in parallel. Additionally, the scikit-learn classifiers will also utilize multithreading, by default using `n_jobs=32`. This is defined in [src/eval/supervised/const.py](src/eval/supervised/const.py).

Some datasets are very large, and for those cases the memory_weight parameter can be used to control the memory usage. This was included due to the OOM issues for larger datasets with high parallelization. If `memory_weight` value is set, the number of `n_jobs` will be divided by `memory_weight` (rounded down). For `n_jobs=8` and `memory_weight=2`, the actual number of jobs will be 4. See [molmuv](config/dataset/clf_ogbg-molmuv.yaml) for an example of `memory_weight` usage.

## Cite 

If you find this work useful in your research, please cite:

```
@article{praski2025benchmarking,
  title={Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning},
  author={Praski, Mateusz and Adamczyk, Jakub and Czech, Wojciech},
  journal={arXiv preprint arXiv:2508.06199},
  year={2025}
}
```

## License

Feel free to use any parts of this project in your research. If you do so, please cite our paper as shown above. Due to the vast number of models used in this project, we're unable to provide a unified license for the entire repository. Please refer to the individual model implementations for their respective licenses.
