# Qdrant reranking search

## Preparation

Make the dataset script executable and run it:

```bash
chmod +x ./download_datasets.sh
./download_datasets.sh
```

This will download the required datasets.

That’s all. To test the application, navigate to the notebooks directory and execute the notebooks.

## High level overview of the modules

```md
project/
├── datasets/
│   ├── wikivoyage-listings-en.csv
│   └── wikivoyage-eu-cities.csv
├── embeddings/
│   ├── dense_embeddings.npy
│   └── etc...
├── src/
│   ├── models.py                   # collection names, model names, enums
│   ├── injest_collection.py        # qdrant collection handlings 
│   └── search.py                   # search strategies (all three of the reranking approaches)
├── notebooks/
│   ├── 01_reranking.ipynb
│   ├── 02_testing-reranking.ipynb

```
