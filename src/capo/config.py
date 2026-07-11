from dataclasses import dataclass, field
from capo.constants import SEED

@dataclass
class DatasetConfig:
    name: str
    source_type: str  # "hf" or "csv"
    source: str       # HF dataset name or local file path
    target_column: str
    feature_columns: list[str] | None = None
    hf_subset: str | None = None


DEFAULT_AGENT_TOOLS: list[str] = ["Skill", "Grep", "Read", "Write", "Bash"]


@dataclass
class AgentGeneratorConfig:
    model_name: str                       # e.g. "claude-sonnet-4-6"
    allowed_tools: list[str] | None = field(default_factory=lambda: list(DEFAULT_AGENT_TOOLS))
    prompt_path: str | None = None
    extra_instructions: str | None = None
    max_iterations: int = 3
    max_turns: int = 10
    permission_mode: str = "acceptEdits"
    preprocessor_key: str | None = None   # auto-derived if None


@dataclass
class ExperimentConfig:
    dataset_name: str
    static_preprocessors: list[str] = field(default_factory=list)
    run_scaled: bool = True
    run_unscaled: bool = False
    n_splits: int = 5
    show_plots: bool = True
    save_csv: bool = True
    random_state: int = SEED
    n_repeats: int = 1
    results_tag: str | None = None
    experiment_name: str | None = None
    continue_experiment: bool = False


DATASET_CONFIGS = {
    "breast": DatasetConfig(
        name="breast",
        source_type="hf",
        source="mstz/breast",
        hf_subset="cancer",
        target_column="is_cancer",
        feature_columns=[
            "clump_thickness",
            "uniformity_of_cell_size",
            "uniformity_of_cell_shape",
            "marginal_adhesion",
            "single_epithelial_cell_size",
            "bare_nuclei",
            "bland_chromatin",
            "normal_nucleoli",
            "mitoses",
        ],
    ),
    "diabetes": DatasetConfig(
        name="diabetes",
        source_type="csv",
        source="data/diabetes/diabetes.csv",
        target_column="Outcome",
        feature_columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """
    helper to get dataset config

    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[dataset_name]