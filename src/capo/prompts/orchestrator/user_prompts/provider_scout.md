You are orchestrating a cloud provider selection task.
Use the cloud-provider-connector agent to select compute for this workload:

{workload_description}
Budget: {budget_constraint}

After the agent responds, return ONLY a JSON object:
{{
  "selected_provider": "<name or null>",
  "selected_instance_type": "<type or null>",
  "instance_specs": {{"gpu": "...", "ram_gb": 0}} or null,
  "estimated_cost_hourly": <number or null>,
  "estimated_cost_total": <number or null>,
  "budget_remaining_or_gap": "<string or null>",
  "connection_method": "<method or null>",
  "connection_steps": ["..."],
  "script_or_template": "<content or null>",
  "rejected_cheaper_options": ["..."],
  "risks_or_missing_config": ["..."]
}}
No text outside the JSON block.