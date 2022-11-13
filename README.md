# EECS595Emoji

## Standardized Virtual Environment and Dependencies:

module load python/3.9.12 cuda/11.6.2
python3 -m venv [env_name]
source ./[env_name]/bin/activate

pip install torch==1.13.0

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip install pandas

pip install tdqm

pip install tdqm

pip install python-csv

pip install matplotlib
