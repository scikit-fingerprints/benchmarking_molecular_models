from src.common.types import SmilesEmbedder

from skfp.fingerprints import ECFPFingerprint, AtomPairFingerprint, TopologicalTorsionFingerprint
from skfp.preprocessing import MolFromSmilesTransformer
from sklearn.pipeline import make_pipeline


class SKFingerprintEmbedder(SmilesEmbedder):
    FINGERPTINT_NAME = None
    
    def __init__(self, fingerprint, raw_name):
        if 'count' in raw_name:
            self._count = True
        else:
            self._count = False
        self._pipeline = make_pipeline(MolFromSmilesTransformer(), fingerprint(count=self._count))
        
    def forward(self, smiles):
        return self._pipeline.transform(smiles)

    @property
    def name(self):
        if not self._count:
            return self.FINGERPTINT_NAME
        return f'{self.FINGERPTINT_NAME}_count'
    
    @property
    def device_used(self):
        return 'cpu'  # SKFP does not use GPU, so we return 'cpu'


class ECFPEmbedder(SKFingerprintEmbedder):
    def __init__(self, raw_name):
        self.FINGERPTINT_NAME = 'ECFP'
        super().__init__(ECFPFingerprint, raw_name) 
        

class AtomPairEmbedder(SKFingerprintEmbedder):
    def __init__(self, raw_name):
        self.FINGERPTINT_NAME = 'AtomPair'
        super().__init__(AtomPairFingerprint, raw_name)
        

class TTEmbedder(SKFingerprintEmbedder):
    def __init__(self, raw_name):
        self.FINGERPTINT_NAME = 'TT'
        super().__init__(TopologicalTorsionFingerprint, raw_name)


def get_embedder(name, **kwargs):
    if 'ecfp' in name.lower():
        return ECFPEmbedder(name)
    elif 'atompair' in name.lower():
        return AtomPairEmbedder(name)
    elif 'tt' in name.lower():
        return TTEmbedder(name)
    else:
        raise ValueError(f'Unknown embedder name: {name}')