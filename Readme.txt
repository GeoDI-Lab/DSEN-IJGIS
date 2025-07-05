# Deep Spatial Evolution Network (DSEN)

Source code of the paper: Gravity-Informed Deep Flow Inference for Spatial Evolution in Panel Data

## 1.Requirements

* `python==3.9.16`
* `pytorch==2.1.1`
* `numpy==1.26.4`
* `geopandas==0.14.2`
* `pandas==2.2.3`
* `matplotlib==3.9.2`
* `seaborn==0.13.2`
* `shapely==2.0.5`
* `contextily==1.6.2`
* `scipy==1.13.1`
* `scikit-learn`
* `geopy==2.4.1`
* `tqdm==4.66.5`

## 2.Dataset

We provide all the necessary data for constructing the geospatial network and training our model. The dataset is organized into following components:

```
____ data
  |____ Random_622
    |____ 42 
        |____ test_nonzero_flow.npy
        |____ train_nonzero_flow.npy
        |____ val_nonzero_flow.npy
  |____ dist_matrix_cbg_index.csv
  |____ edge_index_dist_adj.npy
  |____ edge_weight_dist_adj_dis.npy
  |____ node_feature_snap1_norm.npy 
  |____ node_feature_snap2_norm.npy
```

- **Edge Files:**  
  - `edge_index_dist_adj.npy` contains the edge indices used to construct the geospatial network.
  - `edge_weight_dist_adj_dis.npy` contains the corresponding edge weights.

- **Node Feature Files:**  
  - `node_feature_snap1_norm.npy` and `node_feature_snap2_norm.npy` store the normalized geographic context attributes for each Census Block Group (CBG) across two different snapshots.

- **OD Flow Data:**  
  - The `Random_622` directory holds the origin-destination (OD) flow data. Within it, the subdirectory `42` contains the pre-split training (`train_nonzero_flow.npy`), testing (`test_nonzero_flow.npy`), and validation (`val_nonzero_flow.npy`) sets.


## 3. Running Instructions

This section provides guidance on how to execute the DSEN code using our Jupyter Notebooks.

- **DSEN_demo_code.ipynb:**  
  This notebook offers a step-by-step walkthrough of how to run the DSEN code. It guides you through:
  - Setting up the environment
  - Loading the dataset
  - Executing the main DSEN workflow
  - Training and evaluating the model

- **Visualization_and_table.ipynb**  
  Implements all code for:
  - Converting results into enriched dataframes with geographic attributes.
  - Computing evaluation metrics (CPC, RMSE, Corr, MAPE) over different flow intervals.
  - Comparing DSEN with various baselines and ablation variants.
  - Generating visualizations (bar plots, histograms, box plots) and tables for performance analysis.

Before running the notebooks, ensure that all necessary dependencies are installed and that the file paths are correctly configured.

## 4.References

The implement of some baseline models can refer to the following projects:

https://github.com/dizhu-gis/IIDS-Inferring_Interactions_from_Distribution_Snapshots

https://github.com/scikit-mobility/DeepGravity

https://github.com/susurrant/flow-imputation