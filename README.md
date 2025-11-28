# Algonauts 2025 - Feature Extraction Setup

This guide provides installation and setup instructions for extracting multimodal features (text, video, audio) from the Algonauts 2025 dataset on our SLURM cluster.

## Overview

The code extracts embeddings using:
- **LLAMA3.2-3B** for text features from transcripts
- **VJEPA2** for video frame embeddings
- **Wav2Vec2-BERT** for audio embeddings

These features are cached and can be reused for training brain encoding models.

---

## Installation Steps

### 1. Download UV (pip replacement)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart shell or source ~/.bashrc
```

### 2. Configure Git (if not already done)

```bash
# Generate SSH key if needed
# ssh-keygen -t ed25519 -C "your_email@example.com"

git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

### 3. Download Algonauts Dataset with DataLad

```bash
# Create temporary venv for datalad
cd $SCRATCH
uv venv $SCRATCH/datalad --python=3.12
source $SCRATCH/datalad/bin/activate
uv pip install git-annex
uv pip install datalad

# Download dataset
cd $SCRATCH/
datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git
cd $SCRATCH/algonauts_2025.competitors/
datalad get -r -J8 .

deactivate
```

### 4. Create Main Environment and Install Dependencies

```bash
# Load required modules
module load gcc arrow

# Create virtual environment
cd $SCRATCH
uv venv tribe --python=3.12
source $SCRATCH/tribe/bin/activate

# Install PyTorch (cluster-specific version)
pip install --no-index torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Clone and install algonauts-2025 codebase
cd $SCRATCH/
git clone git@github.com:courtois-neuromod/algonauts-2025.git
cd $SCRATCH/algonauts-2025/data_utils
pip install -e .
cd $SCRATCH/algonauts-2025/modeling_utils
pip install -e .

# Install additional dependencies
pip install transformers moviepy spacy nilearn Levenshtein "huggingface_hub[cli]" julius
```

### 5. Get LLAMA3.2-3B Access from HuggingFace

The text feature extraction requires Meta's LLAMA3.2-3B model.

```bash
# Login to HuggingFace
huggingface-cli login
# Paste your HuggingFace token (create one at https://huggingface.co/settings/tokens)
```

Then visit https://huggingface.co/meta-llama/Llama-3.2-3B in your browser:
1. Click "Request Access"
2. Fill out Meta's usage agreement
3. Wait for approval (typically 5-30 minutes)

Verify access:
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B')"
```

### 6. Set Environment Variables

Add to your `~/.bashrc`:

```bash
# Algonauts environment variables
export SLURM_PARTITION="gpubase_bynode_b3"  # Or your preferred GPU partition
export DATAPATH="$SCRATCH/algonauts_2025.competitors"
export SAVEPATH="$SCRATCH"
```

Then reload:
```bash
source ~/.bashrc
```

**Note:** These environment variables are used by `defaults.py` to configure paths. If not set, defaults to `$SCRATCH/algonauts_2025.competitors` for data and `$SCRATCH` for results.

---

## Running Feature Extraction

### Option 1: Test on Small Subset (Recommended First)

1. Edit `/algonauts-2025/extract_features_only.py` line 22 and uncomment:
   ```python
   "study": {"query": "subject_timeline_index<50"},
   ```

2. Submit job:
   ```bash
   cd $SCRATCH/algonauts-2025
   sbatch run_feature_extraction.sh
   ```

3. Monitor progress:
   ```bash
   squeue -u $USER
   tail -f $SCRATCH/logs/feature_extraction_*.out
   ```

### Option 2: Full Dataset Extraction

1. Ensure the query line in `extract_features_only.py` is commented out
2. Submit job (same command as above)
3. Expect 6-12 hours for completion

---

## Verifying Feature Extraction

After the job completes, verify features were extracted:

```bash
# Check cache directories exist
ls -lh $SCRATCH/cache/algonauts-2025/

# Count cached items per feature type
find $SCRATCH/cache/algonauts-2025/LLAMA3p2_* -name "*.npy" | wc -l
find $SCRATCH/cache/algonauts-2025/VJEPA2_* -name "*.npy" | wc -l
find $SCRATCH/cache/algonauts-2025/Wav2VecBert_* -name "*.npy" | wc -l

# Check total size (expect 5-20GB)
du -sh $SCRATCH/cache/algonauts-2025/
```

Expected structure:
```
$SCRATCH/cache/algonauts-2025/
├── LLAMA3p2_<hash>/
│   ├── meta.json
│   └── item_*.npy files (text embeddings)
├── VJEPA2_<hash>/
│   ├── meta.json
│   └── item_*.npy files (video embeddings)
└── Wav2VecBert_<hash>/
    ├── meta.json
    └── item_*.npy files (audio embeddings)
```

---

## Using Extracted Features for Training

Once features are cached, run training:

```bash
source $SCRATCH/tribe/bin/activate
cd $SCRATCH/algonauts-2025

# Quick test (6 epochs, small data)
python -m algonauts2025.grids.test_run

# Full grid search on SLURM
python -m algonauts2025.grids.run_grid

# Ensemble training
python -m algonauts2025.grids.run_ensemble
```

Cached features are automatically reused - no need to re-extract!

---

## Available GPU Partitions

On our cluster:
- `gpubase_interac` - 8 hours (for quick testing)
- `gpubase_bynode_b2` - 12 hours
- `gpubase_bynode_b3` - 24 hours (recommended for feature extraction)
- `gpubase_bygpu_b3` - 24 hours (GPU-based billing alternative)

Check available partitions:
```bash
sinfo | grep gpu
```

---

## Troubleshooting

**LLAMA3.2 Access Denied:**
- Request access at https://huggingface.co/meta-llama/Llama-3.2-3B
- Create a new HuggingFace token with `read` permissions
- Re-run: `huggingface-cli login`

**GPU Out of Memory:**
- Request larger GPU: `#SBATCH --gres=gpu:h100:1` in `run_feature_extraction.sh`

**Job Timeout:**
- Increase time limit in `run_feature_extraction.sh`: `#SBATCH --time=24:00:00`

**Missing Packages:**
```bash
source $SCRATCH/tribe/bin/activate
cd $SCRATCH/algonauts-2025/data_utils && pip install -e .
cd $SCRATCH/algonauts-2025/modeling_utils && pip install -e .
```

---

## Resource Requirements

- **GPU:** 1x GPU (H100, A100, or V100)
- **RAM:** 64GB recommended
- **GPU Memory:** 25-30GB peak usage
- **Storage:** 5-20GB for cached features
- **Time:** 6-12 hours for full dataset

---

## Files Overview

- `algonauts2025/grids/defaults.py` - Configuration (uses env variables)
- `extract_features_only.py` - Feature extraction script
- `run_feature_extraction.sh` - SLURM submission script
- `$SCRATCH/cache/algonauts-2025/` - Cached features location
- `$SCRATCH/logs/` - Job logs location