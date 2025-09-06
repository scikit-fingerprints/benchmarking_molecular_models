import numpy as np
import paddle.nn as nn
import paddle

from gem_repo.apps.pretrained_compound.ChemRL.GEM.src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from pahelix.utils import load_json_config
from pahelix.model_zoo.gem_model import GeoGNNModel

from tqdm import tqdm
from os.path import join
from hydra.utils import get_original_cwd
from src.common.types import SmilesEmbedder


class GEMEmbedder(SmilesEmbedder):
    def __init__(self, model_path, task):
        task_type = 'regr' if task == 'regression' else 'class'

        config_path = join(get_original_cwd(), "model_wrappers/gem/gem_repo/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json")

        compound_encoder_config = load_json_config(config_path)

        self.transformer = DownstreamTransformFn(is_inference=True)
        self.collator = DownstreamCollateFn(
            atom_names=compound_encoder_config['atom_names'],
            bond_names=compound_encoder_config['bond_names'],
            bond_float_names=compound_encoder_config['bond_float_names'],
            bond_angle_float_names=compound_encoder_config['bond_angle_float_names'],
            task_type=task_type,
            is_inference=True)

        self.compound_encoder = GeoGNNModel(compound_encoder_config)
        self.compound_encoder.set_state_dict(paddle.load(model_path))
        self._dim = self.compound_encoder.graph_dim
        self.norm = nn.LayerNorm(self.compound_encoder.graph_dim)

    def step(self, smile):
        data = self.transformer({"smiles": smile})
        atom_bond_graphs, bond_angle_graphs = self.collator([data])
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()

        _, _, graph_repr = self.compound_encoder(atom_bond_graphs, bond_angle_graphs)
        return self.norm(graph_repr)
    
    def safe_step(self, smile):
        try:
            return self.step(smile).numpy()
        except Exception as e:
            print(f"Error processing SMILES '{smile}': {e}")
            return np.array([float('nan')] * self._dim)
        

    def forward(self, smiles) -> np.ndarray:
        repr = [
            self.safe_step(smile)  # Use safe_step to handle errors gracefully
            for smile in tqdm(smiles)
        ]
        return np.vstack(repr)

    @property
    def name(self):
        return "GEM"
    
    @property
    def device_used(self):
        return "cpu"  # GEM does not use GPU, so we return "cpu"


def get_embedder(model_name, task, **kwargs):
    if model_name.lower() == "gem":
        model_type = "regr" if task == "regression" else "class"
        model_path = join(get_original_cwd(), "model_wrappers/gem/pretrain_models-chemrl_gem", f"{model_type}.pdparams")
        return GEMEmbedder(model_path, task=task)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

