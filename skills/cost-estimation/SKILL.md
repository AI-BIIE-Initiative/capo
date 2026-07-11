---
name: cost-estimation
description: >
 This skill should be used when users need to estimate the cost of cloud workloads such as preprocessing, batch inference, experiments, or protein language model fine-tuning across different providers. 
 Covers :
 (1) generic cost formulas, 
 (2) hardware sizing tradeoffs, 
 (3) runtime estimation, 
 (4) cost optimization guidance 
 (5) links to provider-specific pricing references. 
 Designed to be reusable across Hugging Face Jobs, AWS, Google Cloud, Azure, Lambda, and similar compute providers.
---


# Cost Estimation

**General guidelines:**

```text
Total Cost = (Hours of runtime) × (Cost per hour)
````

**Example calculations:**

**Quick test:**

* Hardware: cpu-basic ($0.10/hour)
* Time: 15 minutes (0.25 hours)
* Cost: $0.03

**Protein data processing:**

* Hardware: l4x1 ($2.50/hour)
* Time: 2 hours
* Cost: $5.00

**Protein model fine-tuning:**

* Hardware: a10g-large ($5/hour)
* Time: 4 hours
* Cost: $20.00

**Cost optimization tips:**

1. Start small - Begin with the lowest-cost hardware that can run the workload, then scale up only if needed
2. Monitor runtime - Set appropriate timeouts
3. Use checkpoints - Resume if job fails
4. Optimize code - Reduce unnecessary compute
5. Choose right hardware - Don't over-provision

## Provider Pricing References

Before answering any cost question, read the reference file for the requested provider using its path relative to this skill's base directory. Do not estimate or guess prices — always read the file first.

| Provider           | Typical use in this project                                | Reference file (read before answering)   |
| ------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| Hugging Face Jobs  | Managed jobs for preprocessing, inference, and fine-tuning | `references/huggingface-jobs-pricing.md` |
| Lambda             | GPU instances for training and fine-tuning                 | `references/lambda-pricing.md`           |
| AWS                | General-purpose cloud compute, EC2 GPU training jobs       | `references/aws-pricing.md`              |
| Google Cloud       | Compute Engine / Vertex AI GPU workloads                   | `references/google-cloud-pricing.md`     |
| Azure              | GPU virtual machines and AML workloads                     | `references/azure-pricing.md`            |

## Guidance

* Prefer official pricing pages over copied values
* Note whether the quoted price is on-demand, reserved, or spot
* Record the region when saving an example price
* Record the exact hardware SKU used for the estimate
* Re-check pricing before running expensive jobs

