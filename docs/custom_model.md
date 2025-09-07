
# Adding Custom Models

---

## Structure

Your custom model should be defined under `model_wrappers/{your_model}`. This directory must contain at least three files: `wrapper.py`, `.python-version`, and `init.sh`. Additionally, please define a Hydra configuration in `config/experiment`.


### .python-version

This file should contain a single line with the Python version you want to use for your model. For example:

```
3.11
```

It will be used to initialize the virtual environment using pyenv.


### wrapper.py

To add a custom model, define a new wrapper class inside `wrapper.py`. The wrapper class should inherit from either [SmilesEmbedder](../src/common/types.py) or [GraphEmbedder](../src/common/types.py), depending on the type of model you are implementing. The class needs to implement the following methods:

- `forward(self, smiles | graphs)`: This method should define the embedding procedure. The input is the full dataset, either in SMILES format or as a list of graphs. Please add batch processing if necessary. The output should be a numpy ndarray of shape (N, D), where N is the number of samples and D is the embedding dimension.
    - If you want your model to support failing on some samples, please return `np.nan * D` for the samples that failed to embed. See [here](../model_wrappers/huggingmolecules/wrapper.py) for an example.
- Property `name`: This property should return the name of the model as a string. If you're using multiple models for the same class, please make sure that the names are unique and match those defined in the Hydra config.

Finally, your `wrapper.py` file should also contain a function `get_embedder()` that returns an instance of your wrapper class. This function is used to initialize the model in the embedding script. It can be as simple as:

```python
def get_embedder(name, **_kwargs):  # <-- **kwargs are important, as this function is called with other parameters, such as task (classification/regression)
    if 'clamp' in name.lower():
        return CLAMPEmbedder()
    raise ValueError(f"Unknown embedder {name}")
```


### init.sh

`init.sh` is used once for every model to initialize the `venv` with all dependencies needed, and optionally to download any model weights. The script is run via [embed_wrapper.sh](../embed_wrapper.sh). Every implementation in our benchmark uses a separate `venv` to avoid dependency conflicts.

The virtual environment is provided via the `embed_wrapper.sh` script and is initialized before `init.sh` is run. You can use the `$INSTALL_DEP` variable as a flag to configure dependency installation only once.

At the end, `init.sh` should provide the environment variable `HYDRA_EXPERIMENT` containing the name of the experiment in `config/experiment/`. This is used to configure the model via Hydra.


### Hydra Configuration

You can use the following template for a simple Hydra config:

```yaml
# @package _global_

defaults:
    - _self_

model_name: your_model_name  # Unique name of the model
```

If you want to provide multiple variants of the model, please refer to examples such as [fingerprints](../config/experiment/fingerprints.yaml) or [huggingmolecules](../config/experiment/huggingmolecules.yaml).


## Running the Model

Once you have defined your model, you can run it using the `embed_wrapper.sh` script to perform embedding on all datasets. By default, all datasets are used; you can modify this by changing the `hydra.sweeper.params.dataset` parameter. See [example here](../config/embed.yaml).

Example command to run the model:

```bash
bash embed_wrapper.sh model_wrappers/your_model
```


This will create embeddings for all datasets and store them in the embedding directory, by default: `data/embedded/dataset_name/model_name.(joblib|json)`. You can change the embedding directory by modifying the [embedding config](../config/embedding/default.yaml).

To evaluate the scores on the datasets using the predefined model grid, please run the following command:

```sh
./run_scoring.sh your_model_name
```


After that, the results will be stored in the `data/meta.db` SQLite database.
