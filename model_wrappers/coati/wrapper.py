import torch

from coati_repo.coati.generative.coati_purifications import embed_smiles
from coati_repo.coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from typing import Optional

from src.common.utils import get_device
from src.common.types import SmilesEmbedder


class CoatiEmbedder(SmilesEmbedder):
    def __init__(self, model_path: str = "s3://terray-public/models/barlow_closed.pkl", device: Optional[str] = None):
        self._device = get_device(device)
        self._encoder, self._tokenizer = load_e3gnn_smiles_clip_e2e(
            freeze=True,
            device=self._device,
            # model parameters to load.
            doc_url=model_path,
        )
        self._tokenizer.on_unknown = "ignore"
        
    def forward(self, smiles):
        with torch.no_grad():
            return torch.stack([
                embed_smiles(s, self._encoder, self._tokenizer)
                for s in smiles
            ]).to('cpu').numpy()

    @property
    def name(self):
        return "coati"
    
    @property
    def device_used(self):
        return self._device.type.split(':')[0]  # Returns 'cuda' or 'cpu' based on the device used


def get_embedder(name, **kwargs) -> SmilesEmbedder:
    return CoatiEmbedder()
