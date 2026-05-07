# WIS Data Science OlmoEarth Pretrain 2026

## Project Purpose
This project is part of the "Projects in Data Science Course 2026 - Geospatial Foundation Models" at the Weizmann Institute of Science (WIS). It focuses on the development and pre-training of state-of-the-art geospatial foundation models, specifically using the **OlmoEarth** foundation model by the Allen Institute as a case study.

## Project Structure
The codebase is organized as follows:

- `olmoearth_pretrain/`: The core library for model development and training.
    - `nn/`: Contains neural network components including FlexiViT, MAE, attention mechanisms, and positional encodings.
    - `data/`: Data loading utilities, including dataloaders, collation, normalization, and visualization.
    - `dataset/`: Logic for parsing and sampling from processed datasets.
    - `dataset_creation/`: Tools and scripts for ingesting raw geospatial data from various sources and converting it to the internal HDF5 format.
    - `evals/`: Evaluation framework for testing model performance (e.g., KNN, linear probing).
    - `train/`: Training modules, loss functions, and masking strategies.
- `data/`:
    - `dataset/`: Local storage for processed HDF5 sample data.
    - `rslearn_dataset_configs/`: JSON configurations for `rslearn` to handle different geospatial data sources.
- `assignments/`: Educational materials and homework (Jupyter notebooks and tests) for the course.
- `WIS_DSP_lib/`: Helper library specifically for the course exercises.
- `flexi_vit.py`: A root-level implementation or variant of the FlexiViT model.

## Used Tools
- **uv**: Fast Python package and environment manager.
- **PyTorch**: Deep learning framework used for all model implementations.
- **rslearn**: Library for geospatial data ingestion and preparation.
- **h5py**: Used for efficient storage and access of large-scale geospatial image tiles.
- **ruff**: Used for linting and code formatting.
- **pytest**: Framework for running automated tests.

## Data Modalities
The project supports a wide range of geospatial data modalities, including:
- **Satellite Imagery**: Sentinel-1, Sentinel-2, Landsat, NAIP.
- **Climate/Weather**: ERA5.
- **Land Cover/Classification**: WorldCover, WorldCereal, CDL, OpenStreetMap (rasterized).
- **Elevation**: SRTM.
- **Socio-economic/Environment**: WorldPop, WRI Canopy Height Map.
- **Ancillary**: Lat/Lon coordinates and timestamps.

Data is typically processed into HDF5 files containing $256 \times 256$ pixel tiles at multiple resolutions, often centered around a Base GSD of 10m.
