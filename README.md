# Deep Spatial Evolution Network (DSEN)

## Overview

This repository contains the official implementation of **"Gravity-Informed Deep Flow Inference for Spatial Evolution Modeling in Panel Data"**, published in the International Journal of Geographical Information Science.

### Abstract

Spatial flows between consecutive distribution snapshots describe how one configuration evolves into the next. Current flow generation models focus on cross-sectional scenarios and neglect the temporal nature of flows. We introduce a **Deep Spatial Evolution Network (DSEN)** to infer panel flows between two snapshots of spatial distributions. DSEN incorporates:

- **Cross-event Context Learner**: Encodes contextual features using parallel Graph Attention Networks
- **Gravity-informed Spatial Evolution Decoder**: Learns latent evolutionary features for flow inference

Using device-level mobile positioning data from the Twin Cities Metropolitan Area, Minnesota, DSEN achieves a **14.0% correlation improvement** and **15.3% error reduction** compared to baselines in inferring human flows during the 2021 Christmas holiday.

## Architecture

DSEN consists of two main components:

### 1. Cross-Event Context Learner (CCL)
- Converts spatial distribution snapshots into geospatial networks
- Uses parallel Graph Attention Networks (GAT₁ and GAT₂) to extract cross-event geographic context embeddings
- Captures spatial dependencies before and after events

### 2. Spatial Evolution Decoder (SED)  
- **MLP₁**: Transforms embeddings into evolutionary features (latent representation of evolution process)
- **MLP₂**: Gravity-informed neural network for flow inference using population changes, distance, and evolutionary features

## Repository Structure

```
DSEN-IJGIS-main/
├── main.py                           # Main training script
├── model/
│   └── DSEN.py                       # DSEN model implementation
├── utils.py                          # Utility functions
├── Optim.py                          # Custom optimizer
├── data/                             # Dataset files
│   ├── edge_index_dist_adj.npy       # Graph edge indices
│   ├── edge_weight_dist_adj_dis.npy  # Graph edge weights  
│   ├── node_feature_snap1_norm.npy   # Node features (before Christmas)
│   ├── node_feature_snap2_norm.npy   # Node features (during Christmas)
│   ├── dist_matrix_cbg_index.csv     # Distance matrix for CBGs
│   └── Random_622/42/                # Train/validation/test splits
│       ├── train_nonzero_flow.npy
│       ├── val_nonzero_flow.npy  
│       └── test_nonzero_flow.npy
├── baselines/                        # Baseline model results
├── ablation_study/                   # Ablation study results
├── DSEN_results/                     # DSEN model outputs
├── DSEN_demo_code.ipynb             # Demo notebook
├── Visualization_and_table.ipynb    # Analysis and visualization
└── LICENSE
```

## Quick Start

### Requirements

Create a conda environment with the required dependencies:

```bash
conda create -n dsen python=3.9.16
conda activate dsen

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install torch-geometric==2.4.0
pip install numpy==1.26.4 pandas==2.2.3 geopandas==0.14.2
pip install matplotlib==3.9.2 seaborn==0.13.2 contextily==1.6.2
pip install scipy==1.13.1 scikit-learn geopy==2.4.1 tqdm==4.66.5
pip install shapely==2.0.5
```

### Option 1: Using Jupyter Notebooks (Recommended for Beginners)

1. **Demo Notebook**: Start with `DSEN_demo_code.ipynb` for a step-by-step walkthrough:
   ```bash
   jupyter notebook DSEN_demo_code.ipynb
   ```
   This notebook covers:
   - Environment setup
   - Data loading and preprocessing
   - Model training and evaluation
   - Results visualization

2. **Analysis Notebook**: Use `Visualization_and_table.ipynb` for detailed analysis:
   - Performance metrics computation
   - Baseline comparisons
   - Ablation study results
   - Flow pattern visualization

### Option 2: Command Line Training

Train the DSEN model directly:

```bash
python main.py --model DSEN \
               --epochs 1500 \
               --batch_size 128 \
               --lr 1e-3 \
               --seed 12345 \
               --test_model_name full_DSEN
```

## Dataset

### Study Area
- **Location**: Twin Cities Metropolitan Area (TCMA), Minnesota, USA
- **Spatial Units**: 2,085 Census Block Groups (CBGs)
- **Event**: Christmas holiday mobility patterns (Dec 20-25, 2021)

### Data Components

1. **Mobile Positioning Data**: 3.74M visitation records from 106,919 individuals
2. **Geographic Context**:
   - **Land Use**: 10 categories (residential, commercial, industrial, etc.)
   - **Points of Interest (POI)**: 9 categories (retail, services, transportation, etc.)  
   - **Demographics**: Census population data
3. **Spatial Networks**: Distance-based fully connected graphs

### Input Features (22-dimensional per CBG)
- Population metrics (3): current population, population change, census population
- Land use areas (10): agriculture, commercial, industrial, institutional, office, water, recreation, residential, transportation, undeveloped
- POI counts (9): agriculture, mining, construction, manufacturing, transportation, wholesale, retail, finance, services


## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{zhu2024gravity,
  title={Gravity-Informed Deep Flow Inference for Spatial Evolution Modeling in Panel Data},
  author={Zhu, Di and Ma, Zhongfu},
  journal={International Journal of Geographical Information Science},
  year={2024},
  publisher={Taylor \& Francis}
}
```


## Contact

- **Di Zhu** (Corresponding Author): [dizhu@umn.edu](mailto:dizhu@umn.edu)
- **Zhongfu Ma**: [ma000523@umn.edu](mailto:ma000523@umn.edu)

**Affiliation**: Department of Geography, Environment and Society, University of Minnesota, Twin Cities


## Related Work

For baseline implementations, refer to:
- [IIDS](https://github.com/dizhu-gis/IIDS-Inferring_Interactions_from_Distribution_Snapshots)
- [DeepGravity](https://github.com/scikit-mobility/DeepGravity)  
- [Flow Imputation](https://github.com/susurrant/flow-imputation)

---

**Keywords**: Spatial Evolution, Human Mobility, Gravity Model, Panel Data, GeoAI, Graph Neural Networks, Flow Inference
