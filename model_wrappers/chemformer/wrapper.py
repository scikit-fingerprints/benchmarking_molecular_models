from omegaconf import OmegaConf
from chemformer_repo.molbart.models.chemformer import Chemformer
from chemformer_repo.molbart.data import SynthesisDataModule
from chemformer_repo.molbart.utils.data_utils import DEFAULT_MAX_SEQ_LEN
from src.common.types import SmilesEmbedder
from src.common.utils import get_device
from os.path import join
from hydra.utils import get_original_cwd


class ChemformerWrapper(SmilesEmbedder):
    def __init__(self, model_path):
        self._model_name = model_path.split("/")[-1].split(".")[0]
        self._device = get_device()
        self._gpu = self._device.type == "cuda"
        self._batch_size = 128
        
        self._config = OmegaConf.create({
            "train_mode": "eval",
            "data_device": "cuda" if self._gpu else "cpu",
            "n_gpus": 1 if self._gpu else 0,
            "vocabulary_path": join(get_original_cwd(), "model_wrappers/chemformer/chemformer_repo/bart_vocab.json"),
            "model_path": join(get_original_cwd(), "model_wrappers/chemformer/weights/", model_path),
            "model_type": "bart",
            "task": "forward_prediction",
            "n_beams": 10,
        })
        
        self._model = Chemformer(
            self._config
        )
        
    def forward(self, smiles):
        print("Performing forward prediction")
        
        smiles_list = smiles.tolist()
        
        datamodule = SynthesisDataModule(
            reactants=smiles_list,
            products=smiles_list,
            tokenizer=self._model.tokenizer,
            batch_size=self._batch_size,
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            dataset_path="",
        )
        datamodule.setup()
        
        outputs = self._model.embed(dataloader=datamodule.full_dataloader())
        return outputs.detach().cpu().numpy()
    
    @property
    def name(self):
        return self._model_name
    
    @property
    def device_used(self):
        return self._device.type.split(':')[0]  # Returns 'cuda' or 'cpu' based on the device used


def get_embedder(name, **kwargs):
    if "chemformer" in name.lower():
        return ChemformerWrapper(name)
    raise ValueError(f"Embedder {name} not found")
