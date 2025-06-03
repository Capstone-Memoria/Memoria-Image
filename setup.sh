#!/bin/bash

python -m venv venv
source venv/bin/activate

# install torch 2.7.0+cu124
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# download the model
mkdir -p models
curl -L https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_V2-SFW.safetensors?download=true -o models/model.safetensors

./start.sh
tail -f app.log