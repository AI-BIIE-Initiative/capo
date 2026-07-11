---
name: google-cloud-pricing
description: Estimate compute cost for GPU workloads on Google Cloud. Pre-computed total GPU cost per hour for on-demand and preemptible instances across common configurations, organized by GPU model.
---

# Google Cloud GPU Pricing Reference

Source: https://cloud.google.com/compute/gpus-pricing — USD, us-central1 region, excluding tax. Prices as of March 2026.

> **Important:** On GCP, GPU costs are billed separately from the underlying VM (vCPUs + memory). The totals below cover GPU cost only. Add VM base cost for the full instance price.

**Use `$/hr GPU cost` directly** — it is pre-computed as `N × $/GPU/hr`. No multiplication needed.

---

## H100 80 GB — A3 series (a3-highgpu-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 80 GB | $14.19 | **$14.19** | $3.69 | **$3.69** |
| 2× | 160 GB | $12.63 | **$25.26** | $2.57 | **$5.14** |
| 4× | 320 GB | $11.84 | **$47.36** | $2.01 | **$8.04** |
| 8× | 640 GB | $11.06 | **$88.48** | $1.45 | **$11.60** |

---

## A100 80 GB — A2 Ultra series (a2-ultragpu-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 80 GB | $4.03 | **$4.03** | $1.57 | **$1.57** |
| 2× | 160 GB | $3.98 | **$7.96** | $1.57 | **$3.14** |
| 4× | 320 GB | $3.95 | **$15.80** | $1.57 | **$6.28** |
| 8× | 640 GB | $3.93 | **$31.44** | $1.57 | **$12.56** |

---

## A100 40 GB — A2 series (a2-highgpu-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 40 GB | $3.67 | **$3.67** | $1.15 | **$1.15** |
| 2× | 80 GB | $3.30 | **$6.60** | $1.15 | **$2.30** |
| 4× | 160 GB | $3.11 | **$12.44** | $1.15 | **$4.60** |
| 8× | 320 GB | $2.93 | **$23.44** | $1.15 | **$9.20** |

---

## L4 24 GB — G2 series (g2-standard-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 24 GB | $0.71 | **$0.71** | $0.22 | **$0.22** |
| 2× | 48 GB | $0.64 | **$1.28** | $0.22 | **$0.44** |
| 4× | 96 GB | $0.60 | **$2.40** | $0.22 | **$0.88** |
| 8× | 192 GB | $0.56 | **$4.48** | $0.22 | **$1.76** |

---

## T4 16 GB — N1 series (n1-standard-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 16 GB | $0.35 | **$0.35** | $0.14 | **$0.14** |
| 2× | 32 GB | $0.35 | **$0.70** | $0.14 | **$0.28** |
| 4× | 64 GB | $0.35 | **$1.40** | $0.14 | **$0.56** |

---

## V100 16 GB — N1 series (n1-standard-*)

| Config | VRAM | $/GPU/hr (on-demand) | $/hr GPU cost (on-demand) | $/GPU/hr (preemptible) | $/hr GPU cost (preemptible) |
|--------|------|----------------------|--------------------------|------------------------|----------------------------|
| 1× | 16 GB | $2.48 | **$2.48** | $0.99 | **$0.99** |
| 2× | 32 GB | $2.48 | **$4.96** | $0.99 | **$1.98** |
| 4× | 64 GB | $2.48 | **$9.92** | $0.99 | **$3.96** |
| 8× | 128 GB | $2.48 | **$19.84** | $0.99 | **$7.92** |

---

## Legacy GPUs (reference only)

| GPU | VRAM | Status | $/GPU/hr (on-demand) | $/GPU/hr (preemptible) |
|-----|------|--------|----------------------|------------------------|
| P100 16GB | 16 GB | Legacy → migrate to A100 | $1.46 | $0.56 |
| P4 8GB | 8 GB | Legacy → migrate to L4 | $0.60 | $0.24 |
| K80 | 12 GB | Deprecated — avoid | — | — |

---

## Committed Use Discounts (CUD)

| Term | Discount off on-demand |
|------|------------------------|
| 1-year | ~30–40% |
| 3-year | ~55–70% |

---

## Cost estimation formula

```
total_cost = hours × $/hr GPU cost  +  hours × VM base cost
```

> VM base cost (vCPUs + memory) is billed separately — check https://cloud.google.com/compute/all-pricing for the specific machine type.

**Example — 8× H100 for 24 h (on-demand, GPU only):**
`24 × $88.48 = $2,123.52` (add VM base cost separately)

**Example — 1× T4 for 100 h (preemptible, GPU only):**
`100 × $0.14 = $14.00`

Preemptible instances can be interrupted with 30-second notice — use for fault-tolerant batch jobs only.
