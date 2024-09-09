#!/bin/sh

pip install gymnasium[atari] matplotlib opencv-python torch tqdm IPython PyYaml
apt-get update
apt-get install -y libgl1-mesa-glx
