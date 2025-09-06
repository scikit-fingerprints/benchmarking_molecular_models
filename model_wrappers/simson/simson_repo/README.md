# SimSon

Pytorch Implementation of SimSon (Simple Contrastive Learning of SMILES for Molecular Property Prediction).

----------------
### Data Directory
You can download datasets and model weights from the following link: https://drive.google.com/drive/folders/11vVMQrNog23Xjtb53CTAJ0loQXdmPwEv?usp=drive_link  
All datasets, except for the pretrained dataset consisting of 1M PubChem entries, are available on Google Drive.
You can download the 1M PubChem dataset directly from https://pubchem.ncbi.nlm.nih.gov
For QM7 dataset, you should download QM7 from MoleculeNet (https://moleculenet.org/datasets-1) and convert mat file to csv format.
<pre><code>
SimSon
├──data
│   ├──prediction
│   │   ├──bace.pth
│   │   ├──bbbp.pth
│   ├──tokenizer
│   │   ├──pubchem_part_tokenizer.json
├──models
│   ├──downstream
│   │   ├──tox21_best_model.pth
│   │   ├──lipophilicity_best_model.pth
│   ├──pretrained
│   │   ├──pretrained_best_model.pth
      .
      .
      .
</code></pre>

### Requirements
* numpy==1.22.4
* requests==2.32.3
* scikit-learn==1.5.0
* scipy==1.13.1
* tokenizers==0.19.1
* torch==2.3.1
* torchaudio==2.3.1
* torchvision==0.18.1
* tqdm==4.66.4
* transformers==4.42.3
* x-transformers==1.31.6

### Run
To finetune and inference all downstream datasets, please refer to the inference.py file or run the following code:
<pre><code>
sh downstrem.sh
</code></pre>

To train the models, run the following codes:
<pre><code>
# pretrain the model
python main.py --task pretraining

# finetune the downstream models (bbbp example)
python main.py --task downstream --data bbbp --criterion bce --lr 5e-5 --num_classes 1

# inference the downstream models (bbbp example)
python main.py --task inference --data bbbp --criterion bce --lr 5e-5 --num_classes 1
</code></pre>




