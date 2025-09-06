from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import logging as log

from src.common.types import SmilesEmbedder
from src.common.utils import batch, get_device
from huggingmolecules import MatFeaturizer, RMatFeaturizer, GroverFeaturizer, MatModel, RMatModel, GroverModel


class HuggingMoleculesEmbedder(SmilesEmbedder):
    FEAT_CLS = None
    MODEL_CLS = None
    OUTPUT_DIM = None
    
    def __init__(self, model_path: str):
        self._model_name = model_path
        self._device = get_device()
        self._featurizer = self.FEAT_CLS.from_pretrained(model_path)
        self._model = self.MODEL_CLS.from_pretrained(model_path).to(self._device)
        self._batch_size = 128
        
    def process_step(self, single_smile):
        batch = self._featurizer([single_smile]).to(self._device)
        with torch.no_grad():
            output = self._model(batch)
        return output

    def process_batch(self, smiles):
        batch = self._featurizer(smiles).to(self._device)
        with torch.no_grad():
            output = self._model(batch) # Returns Torch Tensor of shape [n_samples, dims]
        return output
    
    def safe_batch(self, smiles):
        try:
            return self.process_batch(smiles)
        except Exception as e:
            log.warning(f"Batch failed, processing each smile individually: {e}")
            batch_values = []
            for smile in smiles:
                try:
                    batch_values.append(self.process_step(smile))
                except Exception as e:
                    log.error(f"Failed to process smile '{smile}': {e}")
                    err_array = torch.tensor([float('nan')] * self.OUTPUT_DIM, device=self._device).reshape(1, -1)

                    batch_values.append(err_array)
            return torch.cat(batch_values, dim=0)

    def forward(self, smiles):
        return torch.cat(
            [self.safe_batch(b).detach().cpu() for b in batch(smiles, self._batch_size)],
            dim=0
        ).numpy()
    
    @property
    def name(self):
        return self._model_name
    
    @property
    def device_used(self):
        return "cuda"


class MatEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = MatFeaturizer
    MODEL_CLS = MatModel
    OUTPUT_DIM = 1024

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.__strip_model()

    def __strip_model(self):
        self._model.generator.proj = torch.nn.Identity()

 
class RMatEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = RMatFeaturizer
    MODEL_CLS = RMatModel
    OUTPUT_DIM = 3072

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self._batch_size = 2
        self.__strip_model()

    def __strip_model(self):
        self._model.generator.proj = torch.nn.Identity()


class GroverEmbedder(HuggingMoleculesEmbedder):
    FEAT_CLS = GroverFeaturizer
    MODEL_CLS = GroverModel

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.__strip_model()
        
    def process_batch(self, smiles):
        batch = self._featurizer(smiles).to(self._device)
        atom_ffn_output, bond_ffn_output = self._model(batch)
        return torch.cat([atom_ffn_output, bond_ffn_output], 1)

    def __strip_model(self):
        self._model.mol_atom_from_atom_ffn = torch.nn.Identity()
        self._model.mol_atom_from_bond_ffn = torch.nn.Identity()


def get_embedder(name, **_kwargs) -> HuggingMoleculesEmbedder:
    if 'rmat' in name.lower():
        return RMatEmbedder(name)
    elif 'mat' in name.lower():
        return MatEmbedder(name)
    elif 'grover' in name.lower():
        return GroverEmbedder(name)
    raise ValueError(f'Unknown model name: {name}')
