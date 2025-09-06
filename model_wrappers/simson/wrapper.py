

import numpy as np
import torch

from src.common.types import SmilesEmbedder
from src.common.utils import get_device

from simson_repo.model import Xtransformer_Encoder

from tqdm.auto import tqdm
from tokenizers import Tokenizer
from os.path import join
from hydra.utils import get_original_cwd


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SimSonWrapper(SmilesEmbedder):
    def __init__(self):
        model_path = join(get_original_cwd(), "model_wrappers", "simson", "pretrained_best_model.pth")
        tokenizer_path = join(get_original_cwd(), "model_wrappers", "simson", "pubchem_part_tokenizer.json")
        self._device = get_device()
        
        self._model = Xtransformer_Encoder(args=AttrDict(
            dropout=0.1,
            d_model=768,
            nlayers=2,
            nhead=8,
            dic_size=300,
            max_len=512,
        ))
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device)
        )
        self._model.to(self._device)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
    
    def forward(self, smiles):
        embeddings =  []
        for mol in tqdm(smiles, desc="Encoding SMILES with SimSon"):
            tokens = self._tokenizer.encode(mol).ids
            tokens = torch.tensor(tokens, device=self._device).unsqueeze(0)
            with torch.no_grad():
                embedding = self._model(tokens)
            embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)
    
    @property
    def name(self):
        return "SimSon"
    
    @property
    def device_used(self):
        return self._device.type.split(':')[0]  # Returns 'cpu' or 'cuda'
    

def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "SimSon":
        return SimSonWrapper()
    else:
        raise ValueError(f"Unknown embedder: {name}")