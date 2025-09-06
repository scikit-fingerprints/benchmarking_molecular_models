import selfies as sf
import torch
import logging as log
import traceback

from rdkit import Chem
from os.path import join
from hydra.utils import get_original_cwd
from tqdm.auto import tqdm
from src.common.utils import get_device, batch
from src.common.types import SmilesEmbedder
from abc import ABC
from typing import Optional
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaTokenizer, RobertaModel
from joblib import Parallel, delayed
import multiprocessing
from sentence_transformers import SentenceTransformer
from transformers.modeling_utils import SequenceSummary



def smiles_to_selfies(smiles):
    default_constaints = sf.get_semantic_constraints()
    default_constaints['P-1'] = 6 # molhiv
    default_constaints['Fe'] = 10 # molhiv
    default_constaints['Fe+3'] = 10 # molhiv
    default_constaints['Fe+2'] = 9 # molhiv
    sf.set_semantic_constraints(default_constaints)
    try:
        return sf.encoder(Chem.CanonSmiles(smiles))
    except Exception as e:
        try:
            log.error(f"Error encoding SMILES {smiles}: {e}")
            return sf.encoder(smiles)
        except Exception as e2:
            log.error(f"Error encoding SMILES {smiles} with fallback: {e2}")
            log.error(traceback.format_exc())
            return None



class HuggingFaceSmilesEmbedder(SmilesEmbedder, ABC):
    MODEL_PATH = None
    BATCH_MODE = True
    
    def __init__(self, batch_size: int = 128, model_path: Optional[str] = None, device: Optional[str] = None, init_models: bool = True):
        if model_path is None:
            model_path = self.MODEL_PATH
        self._device = get_device(device)
        self._model_name = model_path.split("/")[-1]
        if init_models:
            self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self._model = None
            self._tokenizer = None
        self._batch_size = batch_size
    
    def _model_batch(self, batch):
        raise NotImplementedError()
    
    def _model_step(self, s):
        raise NotImplementedError()
        
    def forward(self, smiles):
        with torch.no_grad():
            if self.BATCH_MODE:
                outputs = [
                    self._model_batch(s).to('cpu')
                    for s in batch(smiles, n=self._batch_size)
                ]
            else:
                outputs = [
                    self._model_step(s).to('cpu')
                    for s in tqdm(smiles)
                ]
        return torch.cat(outputs).numpy()
    
    @property
    def name(self):
        return self._model_name
    
    @property
    def device_used(self):
        return self._device.type.split(':')[0]
    

class MolFormerEmbedder(HuggingFaceSmilesEmbedder):
    MODEL_PATH = "ibm/MoLFormer-XL-both-10pct"

    def _model_batch(self, batch):
        return self._model(**self._tokenizer(batch, return_tensors="pt", padding=True).to(self._device)).pooler_output


class ChemBERTaEmbedder(HuggingFaceSmilesEmbedder):
    MODEL_PATH = "DeepChem/ChemBERTa-10M-MTR"
    
    def _model_batch(self, batch):
        return self._model(**self._tokenizer(batch, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(self._device)).last_hidden_state[:, 0, :]


class ChemGPTEmbedder(HuggingFaceSmilesEmbedder):
    MODEL_PATH = "ncfrey/ChemGPT-4.7M"
    BATCH_MODE = False
    
    def _model_step(self, s):
        s = smiles_to_selfies(s)
        return self._model(**self._tokenizer(s, return_tensors="pt").to(self._device)).last_hidden_state.mean(dim=1)


class SELFormerEmbedder(HuggingFaceSmilesEmbedder):
    BATCH_MODE = False
    
    def __init__(self, batch_size: int = 128, model_path: Optional[str] = None, device: Optional[str] = None):
        super().__init__(batch_size, model_path, device, init_models=False)
        model_path = join(get_original_cwd(), "model_wrappers/huggingface/model_weights", model_path)
        config = RobertaConfig.from_pretrained(model_path)
        config.output_hidden_states = True
        self._tokenizer = RobertaTokenizer.from_pretrained(join(get_original_cwd(), "model_wrappers/huggingface/selformer_repo/data/RobertaFastTokenizer"))
        self._model = RobertaModel.from_pretrained(model_path, config=config).to(self._device)
        self._joblib_backend = Parallel(n_jobs=multiprocessing.cpu_count())
        
    def forward(self, smiles):
        smiles = list(self._joblib_backend(delayed(smiles_to_selfies)(s) for s in smiles))
        return super().forward(smiles)

    def _model_step(self, selfie):
        token = torch.tensor([self._tokenizer.encode(selfie, add_special_tokens=True, max_length=512, padding=True, truncation=True)]).to(self._device)
        output = self._model(token)

        sequence_out = output[0]
        return torch.mean(sequence_out[0], dim=0)
    

class ChemFMEmbedder(HuggingFaceSmilesEmbedder):
    BATCH_MODE = False
    EOS_TOKEN = "<eos>"
    def __init__(self, model_path: str):
        if '_' in model_path:
            model_path = model_path.split('_')[0]
        
        super().__init__(model_path=model_path, init_models=True)
        self._model = torch.nn.DataParallel(self._model).to(self._device)

    def _model_step(self, s):
        s = s + self.EOS_TOKEN
        inputs = self._tokenizer(s, return_tensors="pt", return_token_type_ids=False).to(self._device)
        outputs = self._model(**inputs)
        return outputs.last_hidden_state[0, -1, :] # EOS token embedding


def get_embedder(name, **_kwargs) -> SmilesEmbedder:
    if 'selformer' in name.lower():
        return SELFormerEmbedder(model_path=name)
    elif 'molformer' in name.lower():
        return MolFormerEmbedder(model_path=name)
    elif 'chemberta' in name.lower():
        return ChemBERTaEmbedder(model_path=name)
    elif 'chemgpt' in name.lower():
        return ChemGPTEmbedder(model_path=name)
    elif 'chemfm' in name.lower():
        return ChemFMEmbedder(model_path=name)
    raise ValueError(f'Unknown model name: {name}')
