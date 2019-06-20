#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.
 
jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=../results \
  --execute ./Feature_visualization_of_Learning_hidden_environment_and_water_solubility.ipynb
  
jupyter nbconvert \
  --ExecutePreprocessor.allow_errors=True \
  --ExecutePreprocessor.timeout=-1 \
  --FilesWriter.build_directory=../results \
  --execute ./Feature_visualization_of_Learning_aromaticity.ipynb
