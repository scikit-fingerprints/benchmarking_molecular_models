from src.common.types import SmilesEmbedder
from src.common.utils import get_device
from clamp_repo.clamp.dataset.encode_compound import ClampEncoder

class CLAMPEmbedder(SmilesEmbedder):
    def __init__(self):
        self._encoder = ClampEncoder(device=get_device())

    def forward(self, smiles):
        return self._encoder.encode(smiles)

    @property
    def name(self):
        return "CLAMP"
    
    @property
    def device_used(self):
        return "cuda"

def get_embedder(name, **_kwargs):
    if 'clamp' in name.lower():
        return CLAMPEmbedder()
    raise ValueError(f"Unknown embedder {name}")