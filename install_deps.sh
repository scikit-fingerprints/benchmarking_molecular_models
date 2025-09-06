#!/bin/bash

# Try normal installation first, then install problematic packages with --no-deps

echo "Attempting normal installation first..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Normal installation failed. Installing with --no-deps..."
    
    while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        echo "Installing with --no-deps: $line"
        pip install --no-deps "$line"
        
        if [ $? -ne 0 ]; then
            echo "Failed to install: $line"
        fi
    done < requirements.txt
else
    echo "Normal installation succeeded!"
fi

echo "Installation process complete!"
