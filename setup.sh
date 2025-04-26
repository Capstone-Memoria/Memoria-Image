# download the model to ./models/model.safetensors
# download the model from https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_V2-SFW.safetensors?download=true

# install the dependencies
pip install -r requirements.txt

# download the model
mkdir -p models
curl -L https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_V2-SFW.safetensors?download=true -o models/model.safetensors