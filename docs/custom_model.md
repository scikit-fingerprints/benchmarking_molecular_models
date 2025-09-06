# Adding custom models

---

## Structure

Your custom model should be defined under `model_wrappers/{your_model}`, and this directory should contain at least three files: `wrapper.py` `.python-version` and `init.sh`. Additionally please define hydra configuration in `config/experiment`

### .python-version

This file should contain single line with the python version you want to use for your model. For example:

```
3.11
```

It will be used to initialize the venv using pyenv.

### wrapper.py

To add a custom model, you need to define a new wrapper class inside the `wrapper.py`. The wrapper class should inherit either [SmilesEmbedder](../src/common/types.py) or [GraphEmbedder](../src/common/types.py) depending on the type of model you are implementing. The class needs to implement the following methods:

- `forward(self, smiles | graphs)`: This method should define the embedding procedure. The input of the function is full dataset, either in SMILES format or as a list of graphs. Please add batch processing if necessary. The output should be a numpy ndarray of shape (N, D) where N is the number of samples and D is the embedding dimension. 
    - If you want your model to support the failing on some of the samples, please return np.nan() * D for the samples that failed to embed. See [here](../model_wrappers/huggingmolecules/wrapper.py) for an example.
- property `name`: This property should return the name of the model as a string. If you're using multiple models for the same class, please make sure that the names are unique, and are the same as defined in the hydra config.

### init.sh

Init is used once for every model to initialize `venv` with all of the dependencies needed, and optionally downloading any model weights. The script is run via the [embed_wrapper.sh](../embed_wrapper.sh) script. Every implementation in our benchmark uses separate `venv` to avoid dependency conflicts. 

Venv is provided via the `embed_wrapper.sh` script, and it's initialized before the `init.sh` script is run. You can use $INSTALL_DEP variable as a flag to configure dependencies installation only once.

At the end, `init.sh` should provide environment variable `HYDRA_EXPERIMENT` contaning name of the experiment in `config/experiment/`. This is used to configure the model via hydra.

### Hydra configuration

You can use following template for a simple hydra config:

```yaml
# @package _global_

defaults:
    - _self_

model_name: your_model_name  # Unique name of the model
```

If you want to provide multiple variants of the model, please refer to examples such as [fingerprints](../config/experiment/fingerprints.yaml) or [huggingmolecules](../config/experiment/huggingmolecules.yaml).

## Running the model

Once you have defined your model, you can run it using the `embed_wrapper.sh` script to perform embedding on all of the datasets. By default all datasets are used, you can modify that by changing the `hydra.sweeper.params.dataset` parameter. See [example here](../config/embed.yaml).

Example command to run the model:

```bash
bash embed_wrapper.sh model_wrappers/your_model
```

This will create embeddings for all datasets and store them in the embedding directory, by default: `data/embedded/dataset_name/model_name.(joblib|json)`. You can change the embedding directory by modifying the [embedding config](../config/embedding/default.yaml).

**_NOTE_**: The extension of the embeddings depends on the Python version. Due to the lack of backward compatibility between certain pandas version, models using Python 3.7 or lower will use json fallback format, while models using Python 3.8 or higher will use joblib format.

To evaluate the scores on the datasets using predefined model grid please run the following command:

```sh
./run_scoring.sh your_model_name
```

After that, the results will be stored in the `data/meta.db` SQLite database.
