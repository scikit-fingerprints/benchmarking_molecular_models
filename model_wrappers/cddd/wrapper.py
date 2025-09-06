import os

from src.common.types import SmilesEmbedder
from cddd.inference import InferenceModel
from hydra.utils import get_original_cwd


class CDDDEmbedder(SmilesEmbedder):
    def __init__(self):
            path = os.path.join(get_original_cwd(), "model_wrappers/cddd/default_model")
            self.__infer_model = InferenceModel(model_dir=path,
                                 use_gpu=False, # This is Python 3.6 so MacOS is out of question
                                 batch_size=512,
                                 cpu_threads=6)
    
    def forward(self, smiles):
        return self.__infer_model.seq_to_emb(smiles)
    
    @property
    def name(self):
        return "CDDD"
    
    @property
    def device_used(self):
        return "cpu"  # CDDD does not use GPU, so we return "cpu"
    
    
def get_embedder(name, **kwargs) -> SmilesEmbedder:
    return CDDDEmbedder()
