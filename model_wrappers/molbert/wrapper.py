import logging as log

from src.common.types import SmilesEmbedder
from hydra.utils import get_original_cwd
from os.path import join
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer


class MolBertEmbedder(SmilesEmbedder):
    def __init__(self, model_path: str):
        # import torch
        # print(torch.version.cuda)
        self._model_name = model_path
        self._embedder = MolBertFeaturizer(join(get_original_cwd(), "model_wrappers/molbert/weights/molbert_100epochs/checkpoints/last.ckpt"), device='cpu')
        log.info(f"Device {self._embedder.device}")
        
    def forward(self, smiles):
        return self._embedder.transform(smiles)[0]
    
    @property
    def name(self):
        return self._model_name
    
    
def get_embedder(name, **_kwagrs) -> MolBertEmbedder:
    return MolBertEmbedder(name)
