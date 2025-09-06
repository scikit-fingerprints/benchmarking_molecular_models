from deepchem.feat import Mol2VecFingerprint
import logging as log
import numpy as np
from src.common.types import SmilesEmbedder


class Mol2VecFingerprintEmbedder(SmilesEmbedder):
    def __init__(self):
        self._embedder = Mol2VecFingerprint() # by default load weights from the article
        
    def forward(self, smiles):
        embeddings = self._embedder.featurize(smiles)
        
        if len(embeddings) == 2:
            # Everything is fine
            return embeddings
        
        log.warning("Some molecules failed")
        
        # Find 1st valid embedding
        shape = None
        for i in range(len(embeddings)):
            if embeddings[i].shape[0] > 0:
                shape = embeddings[i].shape
                break
        
        # Replace all invalid embeddings with NaN embedding
        nan_embed = np.array([float('nan')] * shape[0])
        
        for j in range(len(embeddings)):
            if embeddings[j].shape[0] == 0:
                embeddings[j] = nan_embed
        
        # Stack embeddings into a single array
        return np.stack(embeddings, axis=0)
    
    @property
    def name(self):
        return "mol2vec"
    
    @property
    def device_used(self):
        return "cpu"  # Mol2VecFingerprint does not use GPU, so we return "cpu"

    
def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if name == "mol2vec":
        return Mol2VecFingerprintEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")
