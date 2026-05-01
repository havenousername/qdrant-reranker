# Qdrant reranking search

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
└── presentation/
    └── slides.md
```
