# stimuli_embedding_extraction

```
cd $SCRATCH

curl -LsSf https://astral.sh/uv/install.sh | sh

# Download algonauts dataset
uv venv $SCRATCH/datalad --python=3.12
source $SCRATCH/datalad/bin/activate
uv pip install git-annex
uv pip install datalad
git config --global user.name "XXX"
git config --global user.email XXX
cd $SCRATCH/
datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
cd $SCRATCH/algonauts_2025.competitors/
datalad get -r -J8 .
deactivate

# Tribe

module load gcc arrow
uv venv tribe --python=3.12
source $SCRATCH/tribe/bin/activate
cd $SCRATCH/
git clone https://github.com/facebookresearch/algonauts-2025.git
pip install --no-index torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
cd $SCRATCH/algonauts-2025/data_utils
pip install -e .
cd $SCRATCH/algonauts-2025/modeling_utils
pip install -e .
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius
hf auth login
