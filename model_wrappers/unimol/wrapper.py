import torch

import numpy as np
import logging as log

from src.common.types import SmilesEmbedder
from src.common.utils import get_device, batch
from unimol_tools.predictor import UniMolRepr



class UniMolWrapper(SmilesEmbedder):
    def __init__(self, model_name: str):
        torch.cuda.empty_cache()
        self._model_name = model_name
        use_cuda = False
        # if 'cuda' in get_device().type:
            # use_cuda = True
        # log.info(f"Using device: {get_device()}")
        self._model = UniMolRepr(model_name=model_name, use_gpu=use_cuda, use_cuda=use_cuda)
        self._output_dim = None
        
    def forward_batch(self, smiles_batch):
        log.warning(type(smiles_batch))
        if not isinstance(smiles_batch, list):
            smiles_batch = smiles_batch.tolist()
        unimol_repr = self._model.get_repr(smiles_batch, return_atomic_reprs=False)
        repr_obj = np.array(unimol_repr["cls_repr"])
        self._output_dim = repr_obj.shape[1]
        return repr_obj
    
    def forward_step(self, single_smile):
        batch = [single_smile]
        unimol_repr = self._model.get_repr(batch, return_atomic_reprs=False)
        repr_obj = np.array(unimol_repr["cls_repr"])
        self._output_dim = repr_obj.shape[1]
        return repr_obj
    
    def safe_batch(self, smiles_batch):
        try:
            return self.forward_batch(smiles_batch)
        except Exception as e:
            log.warning(f"Batch failed, processing each smile individually: {e}")
            batch_values = []
            for smile in smiles_batch:
                try:
                    batch_values.append(self.forward_step(smile))
                except Exception as e:
                    log.error(f"Failed to process smile '{smile}': {e}")
                    err_array = np.full((1, self._output_dim), np.nan)
                    batch_values.append(err_array)
            return np.concatenate(batch_values, axis=0)
        
    def forward(self, smiles):
        data = []
        for b in batch(smiles, 128):
            data.append(self.safe_batch(b))
        repr_obj = np.concatenate(data, axis=0)
        return repr_obj
        # log.warning(type(smiles))
        # if not isinstance(smiles, list):
        #     smiles = smiles.tolist()
        # unimol_repr = self._model.get_repr(smiles, return_atomic_reprs=False)
        # repr_obj = np.array(unimol_repr["cls_repr"])
        # return repr_obj
    
    @property
    def name(self):
        return self._model_name
    
    @property
    def device_used(self):
        return get_device().type.split(':')[0]
    

def get_embedder(name: str, **_kwargs) -> SmilesEmbedder:
    if 'unimol' in name:
        return UniMolWrapper(model_name=name)
    else:
        raise ValueError(f"Model name {name} is not recognized as a UniMol model.")
