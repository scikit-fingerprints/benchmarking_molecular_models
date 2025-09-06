
if $INSTALL_DEP; then
    pip install -r requirements.txt
    if [ ! -f "./pretrained_best_model.pth" ]; then
        echo "Please download model from https://drive.google.com/drive/folders/1EZhgntGlPzP9bmUMymaMVYO-DhBWJrxR"
        exit 1
    fi
    if [ ! -f "./pubchem_part_tokenizer.json" ]; then
        echo "Please download tokenizer from https://drive.google.com/drive/folders/1UQU4UfWieQNHv2CIL1IMqPpCyzQH2oeO"
        exit 1
    fi
fi


export HYDRA_EXPERIMENT=simson