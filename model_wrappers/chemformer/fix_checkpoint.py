from glob import glob

import torch
import os


files = glob("weights/*.ckpt")

for file in files:
    print(f"Fixing {file}")
    backup_filename = file + ".backup"
    if os.path.exists(backup_filename):
        print(f"Backup file {backup_filename} already exists. Skipping.")
        continue
    
    os.rename(file, backup_filename)
    try:
        model = torch.load(backup_filename,
                        map_location=lambda storage, loc: storage)
        model["hyper_parameters"]["vocabulary_size"] = model["hyper_parameters"].pop("vocab_size")
        
        # Drop deepspeed dependencies
        if "module" in model:
            model = model["module"]
    except Exception as e:
        print(f"Failed to load {file}. cause: {e}, Skipping.")
        continue
    
    torch.save(model, file)
