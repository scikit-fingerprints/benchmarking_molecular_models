import os
import glob
import json
import joblib
from src.common.types import EmbeddedDataset


def main():
    models = {}
    for model in glob.glob("data/embedded/AMES/*.joblib"):
        data = joblib.load(model)
        if len(data.X.shape) == 1:
            data.X = data.X.reshape(data.y.shape[0], -1)
        models[model] = data.X.shape[1]
    for model in glob.glob("data/embedded/AMES/*.json"):
        data = EmbeddedDataset.deserialize_legacy(model)
        if len(data.X.shape) == 1:
            data.X = data.X.reshape(data.y.shape[0], -1)
        models[model] = data.X.shape[1]    
    
    with open("embedding_size.json", "w") as f:
        json.dump(models, f, indent=4)

if __name__ == "__main__":
    main()