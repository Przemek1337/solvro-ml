# Cocktails Analysis & Clustering

This project focuses on:

-> Exploratory Data Analysis (EDA) of cocktail recipes.

-> Data preprocessing and preparation of features.

-> Clustering (KMeans, DBSCAN, etc.).

-> Evaluation of clustering results, along with basic visualization.

#  Requirements
Python 3.9 or higher (tested on 3.10, but 3.9+ should work).

A package manager, e.g., pip or conda.

# Installation 
1. Clone this repository:

```
git clone https://github.com/<your-org>/cocktails-analysis.git

cd cocktails-analysis
```
2. Install dependencies:
```
pip install -r dependencies.txt
```
# Project Structure
```
solvro-ml/
├── data/
│   ├── cocktail_dataset.json       # Main dataset
├── notebooks/
│   ├── preprocessing.ipynb     # Preprocessing & feature extraction
│   ├── clustering.ipynb        # Clustering (KMeans, DBSCAN, etc.)
│   └── evaluation.ipynb        # Evaluation & cluster visualization
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Loading and cleaning data
│   ├── features.py                # Creating feature matrices
│   ├── clustering.py              # Clustering algorithms
│   ├── evaluation.py              # Evaluation metrics
│   └── visualization.py           # Visualization utilities
├── README.md                      # This file
├── dependencies.txt               # List of required libraries for pip
├── environment.yaml               # Conda environment file (optional)
├── pyproject.toml                 # Alternative dependency management
```
3. Running & Reproducing Experiments

3.1 Running the Jupyter Notebooks

3.1.1 Ensure all dependencies are installed (see Installation section).

3.1.2 Navigate to the notebooks directory and start Jupyter:

```
cd notebooks
jupyter notebook
```

3.1.3 Open and execute the notebooks in order:

-> preprocessing.ipynb – Data cleaning, feature extraction.

-> clustering.ipynb – Clustering with algorithms such as KMeans and DBSCAN.

-> evaluation.ipynb – Assessment of clustering results and visualization of clusters.


