You are orchestrating a model selection task.
Use the model-selector agent to choose the best model for this workload:

{workload_description}

After the agent responds, return ONLY a JSON object:
{{
  "selected_model": "<identifier or null>",
  "fallback_model": "<identifier or null>",
  "estimated_cost": "<string or null>",
  "reasons": ["..."],
  "rejected_candidates": ["..."]
}}
No text outside the JSON block.