---
name: lambda-ai-pricing
description: Estimate compute cost for workloads on Lambda. Includes per-GPU hourly rates and pre-computed total instance rates for on-demand instances and 1-Click Clusters, plus the April 6 2026 price update.
---

# Lambda AI Pricing Reference

Source: https://lambda.ai/pricing — all prices in USD, excluding sales tax.

**Use `$/hr (total)` directly** — no multiplication needed. `$/GPU/hr` is shown for reference only.

---

## On-Demand GPU Instances

### Current rates (effective 00:00 UTC April 6, 2026)

| GPU | Config | VRAM | $/GPU/hr | $/hr (total) |
|-----|--------|------|----------|--------------|
| B200 SXM6 | 1× | 180 GB | $6.99 | **$6.99** |
| B200 SXM6 | 2× | 360 GB | $6.89 | **$13.78** |
| B200 SXM6 | 4× | 720 GB | $6.79 | **$27.16** |
| B200 SXM6 | 8× | 1440 GB | $6.69 | **$53.52** |
| GH200 | 1× | 96 GB | $2.29 | **$2.29** |
| H100 SXM | 1× | 80 GB | $4.29 | **$4.29** |
| H100 SXM | 2× | 160 GB | $4.19 | **$8.38** |
| H100 SXM | 4× | 320 GB | $4.09 | **$16.36** |
| H100 SXM | 8× | 640 GB | $3.99 | **$31.92** |
| H100 PCIe | 1× | 80 GB | $3.29 | **$3.29** |
| A100 SXM 80 GB | 8× | 640 GB | $2.79 | **$22.32** |
| A100 SXM 40 GB | 1× | 40 GB | $1.99 | **$1.99** |
| A100 SXM 40 GB | 8× | 320 GB | $1.99 | **$15.92** |
| A100 PCIe 40 GB | 1× | 40 GB | $1.99 | **$1.99** |
| A100 PCIe 40 GB | 2× | 80 GB | $1.99 | **$3.98** |
| A100 PCIe 40 GB | 4× | 160 GB | $1.99 | **$7.96** |
| A6000 PCIe | 1× | 48 GB | $1.09 | **$1.09** |
| A6000 PCIe | 2× | 96 GB | $1.09 | **$2.18** |
| A6000 PCIe | 4× | 192 GB | $1.09 | **$4.36** |
| A10 | 1× | 24 GB | $1.29 | **$1.29** |
| RTX 6000 | 1× | 24 GB | $0.69 | **$0.69** |
| V100 | 8× | 128 GB | $0.79 | **$6.32** |

---

### Price change — old → new (effective April 6, 2026)

| GPU | Config | Old $/GPU/hr | Old $/hr (total) | New $/GPU/hr | New $/hr (total) | Δ $/hr |
|-----|--------|-------------|-----------------|-------------|-----------------|--------|
| B200 SXM | 1× | $6.08 | $6.08 | $6.99 | $6.99 | +$0.91 |
| B200 SXM | 2× | $5.97 | $11.94 | $6.89 | $13.78 | +$1.84 |
| B200 SXM | 4× | $5.85 | $23.40 | $6.79 | $27.16 | +$3.76 |
| B200 SXM | 8× | $5.74 | $45.92 | $6.69 | $53.52 | +$7.60 |
| GH200 | 1× | $1.99 | $1.99 | $2.29 | $2.29 | +$0.30 |
| H100 SXM | 1× | $3.78 | $3.78 | $4.29 | $4.29 | +$0.51 |
| H100 SXM | 2× | $3.67 | $7.34 | $4.19 | $8.38 | +$1.04 |
| H100 SXM | 4× | $3.55 | $14.20 | $4.09 | $16.36 | +$2.16 |
| H100 SXM | 8× | $3.44 | $27.52 | $3.99 | $31.92 | +$4.40 |
| H100 PCIe | 1× | $2.86 | $2.86 | $3.29 | $3.29 | +$0.43 |
| A100 SXM 80 GB | 8× | $2.06 | $16.48 | $2.79 | $22.32 | +$5.84 |
| A100 SXM 40 GB | 1× | $1.48 | $1.48 | $1.99 | $1.99 | +$0.51 |
| A100 SXM 40 GB | 8× | $1.48 | $11.84 | $1.99 | $15.92 | +$4.08 |
| A100 PCIe 40 GB | 1× | $1.48 | $1.48 | $1.99 | $1.99 | +$0.51 |
| A100 PCIe 40 GB | 2× | $1.48 | $2.96 | $1.99 | $3.98 | +$1.02 |
| A100 PCIe 40 GB | 4× | $1.48 | $5.92 | $1.99 | $7.96 | +$2.04 |
| A6000 PCIe | 1× | $0.92 | $0.92 | $1.09 | $1.09 | +$0.17 |
| A6000 PCIe | 2× | $0.92 | $1.84 | $1.09 | $2.18 | +$0.34 |
| A6000 PCIe | 4× | $0.92 | $3.68 | $1.09 | $4.36 | +$0.68 |
| A10 | 1× | $0.86 | $0.86 | $1.29 | $1.29 | +$0.43 |
| RTX 6000 | 1× | $0.58 | $0.58 | $0.69 | $0.69 | +$0.11 |
| V100 | 8× | $0.63 | $5.04 | $0.79 | $6.32 | +$1.28 |

---

## 1-Click Clusters (per GPU per hour — cluster size varies)

| System | $/GPU/hr |
|--------|----------|
| NVIDIA HGX B200 (16–2,000+ GPUs) | $4.62 |
| NVIDIA H100 | $2.76 |

> Total = `num_gpus × $/GPU/hr`. Cluster size is variable; multiply by your actual GPU count.

Reserved capacity (1-, 2-, 3-year terms): contact Lambda sales.

---

## Cost estimation formula

```
total_cost = hours × $/hr (total)
```

**Example — 8× H100 SXM for 72 h:**
`72 × $31.92 = $2,298.24`
