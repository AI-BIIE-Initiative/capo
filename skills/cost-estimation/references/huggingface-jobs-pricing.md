---
name: huggingface-jobs-pricing
description: Estimate compute cost for GPU workloads on HuggingFace. Total instance hourly rates for Spaces, Inference Endpoints (AWS & GCP), and Training Jobs, organized by GPU model.
---

# HuggingFace GPU Pricing Reference

Source: https://huggingface.co/pricing — USD per hour, excluding tax. Prices as of March 2026.

**Use `$/hr (total)` directly** — it is the full instance cost. No multiplication needed.

---

## Spaces Hardware

Used for hosted ML demos and dev inference.

| GPU | Config | VRAM | $/hr (total) | $/GPU/hr |
|-----|--------|------|--------------|----------|
| T4 | 1× (small) | 16 GB | **$0.40** | $0.40 |
| T4 | 1× (medium) | 16 GB | **$0.60** | $0.60 |
| L4 | 1× | 24 GB | **$0.80** | $0.80 |
| L4 | 4× | 96 GB | **$3.80** | $0.95 |
| L40S | 1× | 48 GB | **$1.80** | $1.80 |
| L40S | 4× | 192 GB | **$8.30** | $2.08 |
| L40S | 8× | 384 GB | **$23.50** | $2.94 |
| A10G | 1× (small) | 24 GB | **$1.00** | $1.00 |
| A10G | 1× (large) | 24 GB | **$1.50** | $1.50 |
| A10G | 2× (large) | 48 GB | **$3.00** | $1.50 |
| A10G | 4× (large) | 96 GB | **$5.00** | $1.25 |
| A100 | 1× | 80 GB | **$2.50** | $2.50 |
| A100 | 4× | 320 GB | **$10.00** | $2.50 |
| A100 | 8× | 640 GB | **$20.00** | $2.50 |
| ZeroGPU | dynamic (H200) | 70 GB | **free** | — |

---

## Inference Endpoints — AWS

| GPU | Config | VRAM | $/hr (total) | $/GPU/hr |
|-----|--------|------|--------------|----------|
| T4 | 1× | 14 GB | **$0.50** | $0.50 |
| T4 | 4× | 56 GB | **$3.00** | $0.75 |
| L4 | 1× | 24 GB | **$0.80** | $0.80 |
| L4 | 4× | 96 GB | **$3.80** | $0.95 |
| L40S | 1× | 48 GB | **$1.80** | $1.80 |
| L40S | 4× | 192 GB | **$8.30** | $2.08 |
| L40S | 8× | 384 GB | **$23.50** | $2.94 |
| A10G | 1× | 24 GB | **$1.00** | $1.00 |
| A10G | 4× | 96 GB | **$5.00** | $1.25 |
| A100 | 1× | 80 GB | **$2.50** | $2.50 |
| A100 | 2× | 160 GB | **$5.00** | $2.50 |
| A100 | 4× | 320 GB | **$10.00** | $2.50 |
| A100 | 8× | 640 GB | **$20.00** | $2.50 |
| H100 | 1× | 80 GB | **$4.50** | $4.50 |
| H100 | 2× | 160 GB | **$9.00** | $4.50 |
| H100 | 4× | 320 GB | **$18.00** | $4.50 |
| H100 | 8× | 640 GB | **$36.00** | $4.50 |
| H200 | 1× | 141 GB | **$5.00** | $5.00 |
| H200 | 2× | 282 GB | **$10.00** | $5.00 |
| H200 | 4× | 564 GB | **$20.00** | $5.00 |
| H200 | 8× | 1128 GB | **$40.00** | $5.00 |
| B200 | 1× | 179 GB | **$9.25** | $9.25 |
| B200 | 2× | 358 GB | **$18.50** | $9.25 |
| B200 | 4× | 716 GB | **$37.00** | $9.25 |
| B200 | 8× | 1432 GB | **$74.00** | $9.25 |

---

## Inference Endpoints — GCP

| GPU | Config | VRAM | $/hr (total) | $/GPU/hr |
|-----|--------|------|--------------|----------|
| T4 | 1× | 16 GB | **$0.50** | $0.50 |
| L4 | 1× | 24 GB | **$0.70** | $0.70 |
| L4 | 4× | 96 GB | **$3.80** | $0.95 |
| A100 | 1× | 80 GB | **$3.60** | $3.60 |
| A100 | 2× | 160 GB | **$7.20** | $3.60 |
| A100 | 4× | 320 GB | **$14.40** | $3.60 |
| A100 | 8× | 640 GB | **$28.80** | $3.60 |
| H100 | 1× | 80 GB | **$10.00** | $10.00 |
| H100 | 2× | 160 GB | **$20.00** | $10.00 |
| H100 | 4× | 320 GB | **$40.00** | $10.00 |
| H100 | 8× | 640 GB | **$80.00** | $10.00 |

---

## Inference Endpoints — CPU Instances

### AWS (Intel Sapphire Rapids)

| vCPUs | Memory | $/hr (total) |
|-------|--------|--------------|
| 1 | 2 GB | $0.03 |
| 2 | 4 GB | $0.07 |
| 4 | 8 GB | $0.13 |
| 8 | 16 GB | $0.27 |
| 16 | 32 GB | $0.54 |

### AWS (Intel Sapphire Rapids — overcommit)

| vCPUs | Memory | $/hr (total) |
|-------|--------|--------------|
| 16 | 32 GB | $0.01 |

### Azure (Intel Xeon)

| vCPUs | Memory | $/hr (total) |
|-------|--------|--------------|
| 1 | 2 GB | $0.06 |
| 2 | 4 GB | $0.12 |
| 4 | 8 GB | $0.24 |
| 8 | 16 GB | $0.48 |

### GCP (Intel Sapphire Rapids)

| vCPUs | Memory | $/hr (total) |
|-------|--------|--------------|
| 1 | 2 GB | $0.05 |
| 2 | 4 GB | $0.10 |
| 4 | 8 GB | $0.20 |
| 8 | 16 GB | $0.40 |

---

## Inference Endpoints — Accelerator Instances

### AWS (Inf2 Neuron)

| Topology | Accelerator Memory | $/hr (total) |
|----------|--------------------|--------------|
| x1 | 14.5 GB | $0.75 |
| x1-large | 124 GB | $1.95 |
| x12 | 760 GB | $12.00 |

### GCP (TPU v5e)

| Topology | Accelerator Memory | $/hr (total) |
|----------|--------------------|--------------|
| 1x1 | 16 GB | $1.20 |
| 2x2 | 64 GB | $4.75 |
| 2x4 | 128 GB | $9.50 |

---

## Cost estimation formula

```
total_cost = hours × $/hr (total)
```

**Example — 8× H100 AWS endpoint for 10 h:**
`10 × $36.00 = $360.00`

**Example — 1× A100 Space for 48 h:**
`48 × $2.50 = $120.00`
