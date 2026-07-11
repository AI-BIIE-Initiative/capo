# Core Papers — Dimensionality Reduction

## t-SNE
van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE.
*Journal of Machine Learning Research*, 9, 2579–2605.
https://jmlr.org/papers/v9/vandermaaten08a.html

Key: stochastic, perplexity-sensitive, no transform on new data, axes are meaningless.

## UMAP
McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection.
arXiv:1802.03426. https://arxiv.org/abs/1802.03426
Docs: https://umap-learn.readthedocs.io

Key: fast, scalable, supports transform() on new data. UMAP for clustering: https://umap-learn.readthedocs.io/en/latest/clustering.html

## PCA and TruncatedSVD
scikit-learn decomposition: https://scikit-learn.org/stable/modules/decomposition.html

Key: TruncatedSVD avoids dense centering — use for sparse inputs (one-hot, k-mer).

## ESM2 Protein Language Models
Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure.
*Science*, 379(6637), 1123–1130. https://doi.org/10.1126/science.ade2574
GitHub: https://github.com/facebookresearch/esm
