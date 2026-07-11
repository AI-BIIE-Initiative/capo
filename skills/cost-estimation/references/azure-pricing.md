---
name: azure-ml-pricing
description: Estimate compute cost for GPU workloads on Azure Machine Learning. Pre-computed total VM hourly rates (on-demand and spot) for on-demand and spot instances, organized by GPU model.
---

# Azure ML GPU Pricing Reference

Source: https://azure.microsoft.com/en-us/pricing/details/machine-learning/ — USD, East US region, excluding tax. Prices as of March 2026.

**Use `$/VM/hr (on-demand)` or `$/VM/hr (spot)` directly** — both are pre-computed total instance costs. No multiplication needed.

---

## H100 80 GB (SXM)

| VM Size | GPUs | $/VM/hr (on-demand) | $/GPU/hr | $/VM/hr (spot) |
|---------|------|---------------------|----------|----------------|
| Standard_NC40ads_H100_v5 | 1× | **$6.98** | $6.98 | **$2.49** |
| Standard_NC80adis_H100_v5 | 2× | **$13.96** | $6.98 | **$4.98** |
| Standard_ND96isr_H100_v5 | 8× | **$98.32** | $12.29 | **$18.16** |

> ND96isr is NVLink-connected (SXM5, HBM3, 3.35 TB/s); higher $/GPU reflects NVLink + InfiniBand fabric.

---

## A100 80 GB (PCIe)

| VM Size | GPUs | $/VM/hr (on-demand) | $/GPU/hr | $/VM/hr (spot) |
|---------|------|---------------------|----------|----------------|
| Standard_NC24ads_A100_v4 | 1× | **$3.67** | $3.67 | **$0.74** |
| Standard_NC48ads_A100_v4 | 2× | **$7.35** | $3.68 | **$1.48** |
| Standard_NC96ads_A100_v4 | 4× | **$14.69** | $3.67 | **$2.96** |

---

## V100 16 GB (PCIe)

| VM Size | GPUs | $/VM/hr (on-demand) | $/GPU/hr | $/VM/hr (spot) |
|---------|------|---------------------|----------|----------------|
| Standard_NC12s_v3 | 2× | **$6.12** | $3.06 | **$1.14** |
| Standard_NC24s_v3 | 4× | **$12.24** | $3.06 | **$2.28** |

---

## T4 16 GB (PCIe)

> $/GPU/hr varies because vCPU/memory differ per VM; GPU is fixed at 16 GB T4.

| VM Size | vCPUs | GPUs | $/VM/hr (on-demand) | $/GPU/hr | $/VM/hr (spot) |
|---------|-------|------|---------------------|----------|----------------|
| Standard_NC4as_T4_v3 | 4 | 1× | **$0.53** | $0.53 | **$0.19** |
| Standard_NC8as_T4_v3 | 8 | 1× | **$0.75** | $0.75 | **$0.28** |
| Standard_NC16as_T4_v3 | 16 | 1× | **$1.20** | $1.20 | **$0.45** |
| Standard_NC64as_T4_v3 | 64 | 4× | **$4.35** | $1.09 | **$1.60** |

---

## M60 (Visualization / NVv3)

| VM Size | GPUs | $/VM/hr (on-demand) | $/GPU/hr | $/VM/hr (spot) |
|---------|------|---------------------|----------|----------------|
| Standard_NV6 | 1× | **$1.14** | $1.14 | **$0.21** |
| Standard_NV12 | 2× | **$2.28** | $1.14 | **$0.84** |

---

## Quick comparison (on-demand $/VM/hr)

| GPU | Config | $/VM/hr (on-demand) | Notes |
|-----|--------|---------------------|-------|
| H100 80 GB | 1× | $6.98 | |
| H100 80 GB | 2× | $13.96 | |
| H100 80 GB | 8× | $98.32 | NVLink node |
| A100 80 GB | 1× | $3.67 | |
| A100 80 GB | 4× | $14.69 | |
| V100 16 GB | 2× | $6.12 | |
| T4 16 GB | 1× | $0.53 – $1.20 | Scales with CPU |
| M60 | 1× | $1.14 | Visualization only |

---

## Cost estimation formula

```
total_cost = hours × $/VM/hr (on-demand or spot)
```

**Example — 4× A100 for 48 h (on-demand):**
`48 × $14.69 = $705.12`

**Example — 8× H100 for 24 h (spot):**
`24 × $18.16 = $435.84`

Spot prices fluctuate — budget with 2–3× spot as a safe upper bound.
