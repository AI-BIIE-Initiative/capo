# Biological Validation of Clusters

Internal clustering metrics (silhouette, Davies-Bouldin) measure geometric quality.
They do not measure biological usefulness. Always validate biologically.

## Checklist

A cluster is useful if at least some of the following hold:

**Structural coherence**
- Sequences within the cluster have consistent length profiles
- Residue composition is coherent (not mixing structured and disordered sequences)
- Pairwise edit distance within cluster is lower than across clusters

**Label coherence**
- Binding labels (binder / non-binder) are enriched in specific clusters — not randomly distributed
- If labels are mixed within a cluster, verify whether biology suggests overlap (e.g. promiscuous binders)

**Species enrichment**
- Each cluster shows enriched representation of one or a few species
- If all clusters have identical species composition, the features may not capture taxonomy

**Mutation patterns**
- Clusters from deep mutational scanning should show position-specific mutation enrichment
- Run `cluster_profiles._mutation_heatmap()` to visualise per-position residue entropy per cluster

**Split integrity**
- Cluster-aware splits maintain class balance across train/val/test
- `check_cluster_leakage()` returns 0 for both `sequences_in_multiple_splits` and `clusters_spanning_splits`

## Red flags

| Warning | Interpretation |
|---------|---------------|
| Single cluster ≥ 80% of data | Parameters too permissive — increase `min_cluster_size` or decrease K |
| All points labeled noise (-1) | Parameters too strict, or data is one dense blob |
| Clusters perfectly match label | Cluster is the label — useful for calibration but not for split creation |
| Silhouette < 0 | Cluster assignments are worse than random — try different algorithm or space |
| Noise fraction > 20% | Many outlier sequences — inspect them rather than discarding |
