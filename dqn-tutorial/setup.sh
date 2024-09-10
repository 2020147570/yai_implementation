#!/bin/sh

pip install gymnasium[atari] gymnasium[accept-rom-license] matplotlib numpy opencv-python torch tqdm IPython PyYaml
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
