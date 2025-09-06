import numpy as np
import warnings

from hydra.utils import get_original_cwd
from mol_r_repo.src.featurizer import MolEFeaturizer
from src.common.types import SmilesEmbedder
from src.common.utils import batch


class MolREmbedder(SmilesEmbedder):
    def __init__(self, variant, path, batch_size: int = 128):
        self._model = MolEFeaturizer(path, gpu=None)
        self._batch_size = batch_size
        self._variant = variant
        
    def batch_step(self, smiles):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.transform(smiles)[0]
    
    def forward(self, smiles):
        outputs = [
            self.batch_step(s)
            for s in batch(smiles, n=self._batch_size)
        ]
        return np.vstack(outputs)
    
    @property
    def name(self):
        return f"mol_r_{self._variant}"
    
    @property
    def device_used(self):
        return "cpu"  # Mol-R does not use GPU, so we return "cpu"
    

def get_embedder(name, **_kwargs):
    if "mol_r" in name.lower():
        variant = name[6:]
        model_path = f"{get_original_cwd()}/model_wrappers/mol-r/mol_r_repo/saved/{variant}"
        
        return MolREmbedder(path=model_path, variant=variant, batch_size=_kwargs.get("batch_size", 128))
    else:
        raise ValueError(f"Unknown embedder name: {name}")