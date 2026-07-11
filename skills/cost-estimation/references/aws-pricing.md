---
name: aws-pricing
description: Estimate compute cost for GPU workloads on Amazon EC2. Pre-computed total instance hourly rates (on-demand and spot) for common GPU configurations, organized by GPU model.
---

# AWS EC2 GPU Pricing Reference

Source: https://aws.amazon.com/ec2/pricing/on-demand/ — USD, us-east-1 region, excluding tax. Prices as of March 2026.

> On AWS, GPU + vCPU + memory are bundled in the instance price — no separate GPU billing.

**Use `$/hr (total, on-demand)` or `Spot $/hr (total)` directly** — both are pre-computed full instance costs. No multiplication needed.

---

## H100 80 GB — P5 family

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| p5.4xlarge | 1× | **$6.88** | $6.88 | $6.87 | **$6.87** |
| p5.48xlarge | 8× | **$55.04** | $6.88 | $3.78 | **$30.24** |

---

## A100 40 GB — P4 family

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| p4d.24xlarge | 8× | **$21.96** | $2.75 | $1.35 | **$10.80** |

---

## V100 16 GB — P3 family

> $/GPU/hr varies with vCPU allocation per instance size.

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| p3.2xlarge | 1× | **$3.06** | $3.06 | $0.39 | **$0.39** |
| p3.8xlarge | 4× | **$12.24** | $3.06 | $0.42 | **$1.68** |

---

## L40S 48 GB — G6e family

> $/GPU/hr varies with vCPU allocation per instance size.

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| g6e.xlarge | 1× | **$1.86** | $1.86 | $1.04 | **$1.04** |
| g6e.2xlarge | 1× | **$2.24** | $2.24 | $1.02 | **$1.02** |

---

## A10G 24 GB — G5 family

> $/GPU/hr varies with vCPU allocation per instance size.

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| g5.xlarge | 1× | **$1.01** | $1.01 | $0.41 | **$0.41** |
| g5.2xlarge | 1× | **$1.21** | $1.21 | $0.48 | **$0.48** |
| g5.4xlarge | 1× | **$1.64** | $1.64 | $0.65 | **$0.65** |

---

## T4 16 GB — G4dn family

> $/GPU/hr varies with vCPU allocation per instance size.

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| g4dn.xlarge | 1× | **$0.53** | $0.53 | $0.21 | **$0.21** |
| g4dn.2xlarge | 1× | **$0.75** | $0.75 | $0.28 | **$0.28** |

---

## Radeon Pro V520 8 GB — G4ad family (AMD)

| Instance | GPUs | $/hr (total, on-demand) | $/GPU/hr | Spot $/GPU/hr | Spot $/hr (total) |
|----------|------|------------------------|----------|---------------|-------------------|
| g4ad.xlarge | 1× | **$0.38** | $0.38 | $0.12 | **$0.12** |
| g4ad.2xlarge | 1× | **$0.54** | $0.54 | $0.27 | **$0.27** |

---

## Quick comparison (on-demand $/hr total)

| GPU | VRAM | Instance | $/hr (total, on-demand) | Notes |
|-----|------|----------|------------------------|-------|
| H100 | 80 GB | p5.4xlarge | $6.88 | 1 GPU |
| H100 | 80 GB | p5.48xlarge | $55.04 | 8 GPUs |
| A100 | 40 GB | p4d.24xlarge | $21.96 | 8 GPUs |
| V100 | 16 GB | p3.2xlarge | $3.06 | 1 GPU |
| V100 | 16 GB | p3.8xlarge | $12.24 | 4 GPUs |
| L40S | 48 GB | g6e.xlarge | $1.86 | 1 GPU |
| A10G | 24 GB | g5.xlarge | $1.01 | 1 GPU |
| T4 | 16 GB | g4dn.xlarge | $0.53 | 1 GPU |

> AWS reduced P5 and P4 pricing by up to 45% and 33% respectively in June 2025.

---

## Cost estimation formula

```
total_cost = hours × $/hr (total, on-demand)
             — or —
total_cost = hours × Spot $/hr (total)
```

**Example — 8× H100 (p5.48xlarge) for 24 h (on-demand):**
`24 × $55.04 = $1,320.96`

**Example — 8× H100 (p5.48xlarge) for 24 h (spot):**
`24 × $30.24 = $725.76`

Spot instances can be interrupted — use for checkpointed or fault-tolerant training jobs only.
