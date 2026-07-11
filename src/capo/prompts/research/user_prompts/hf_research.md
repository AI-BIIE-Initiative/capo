HuggingFace Hub research for one PLM fine-tuning run.

run.model_id:           {model_id}
run.fine_tune_strategy: {fine_tune_strategy}
run.dataset_ref:        {dataset_ref}

{task_context}
Follow the procedure and schema in the system prompt. Stay under 8 shell calls.
Use `hf` CLI for searches and metadata; `curl` only for raw README files.
Every `hf_id` MUST be Hub-verified in this session — emit "unknown" rather
than inventing an ID. Emit ONE JSON object — no prose, no fences.
