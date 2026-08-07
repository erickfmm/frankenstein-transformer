"""Streamlit web interface for schema-driven YAML configuration building.

Provides a multi-tab web application that dynamically renders form fields
from the training configuration JSON Schema, generates YAML output, and
offers CLI command construction with background execution via nohup.
Supports English and Spanish UI localization.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import streamlit as st
import yaml

# Constants
SCHEMA_PATH = Path(__file__).parent.parent / "schema.yaml"
AVAILABLE_COMMANDS = [
    {"id": "train", "name": "Train", "description": "Run main training"},
    {"id": "deploy", "name": "Deploy", "description": "Convert checkpoint to deployment artifacts"},
    {"id": "quantize", "name": "Quantize", "description": "Export checkpoint in quantized deployment format"},
    {"id": "infer", "name": "Infer", "description": "Run deployed model inference"},
    {"id": "sbert-train", "name": "SBERT Train", "description": "Train SBERT model"},
    {"id": "sbert-infer", "name": "SBERT Infer", "description": "Run SBERT inference tasks"},
]

LANG_EN = "en"
LANG_ES = "es"

LANGUAGE_LABELS = {
    LANG_EN: "English",
    LANG_ES: "Español",
}

UI_STRINGS = {
    "required": {
        "en": "Required",
        "es": "Obligatorio",
    },
    "language_selector": {
        "en": "Language",
        "es": "Idioma",
    },
    "parameter_group": {
        "en": "Parameter group",
        "es": "Grupo de parámetros",
    },
    "optimizer_configuration": {
        "en": "Optimizer Configuration",
        "es": "Configuración del Optimizador",
    },
    "parameter_group_info": {
        "en": "The following parameters are available for this optimizer. Prefix each parameter with the optimizer class name (e.g., 'adamw-lr_embeddings').",
        "es": "Los siguientes parámetros están disponibles para este optimizador. Prefija cada parámetro con el nombre de la clase del optimizador (por ejemplo, 'adamw-lr_embeddings').",
    },
    "embeddings_group": {
        "en": "Embeddings",
        "es": "Embeddings",
    },
    "embeddings_caption": {
        "en": "Parameters controlling token embedding matrix optimization.",
        "es": "Parámetros que controlan la optimización de la matriz de embeddings de tokens.",
    },
    "normalization_group": {
        "en": "Normalization Layers",
        "es": "Capas de Normalización",
    },
    "normalization_caption": {
        "en": "Parameters for layer normalization and other normalization layers.",
        "es": "Parámetros para layer normalization y otras capas de normalización.",
    },
    "attention_group": {
        "en": "Attention Layers",
        "es": "Capas de Atención",
    },
    "attention_caption": {
        "en": "Parameters for attention mechanism optimization (all attention variants).",
        "es": "Parámetros para la optimización de mecanismos de atención (todas las variantes).",
    },
    "other_group": {
        "en": "Other Parameters",
        "es": "Otros Parámetros",
    },
    "other_caption": {
        "en": "Parameters for remaining model components (FFN, routing, etc.).",
        "es": "Parámetros para los componentes restantes del modelo (FFN, routing, etc.).",
    },
}


def load_schema() -> Dict[str, Any]:
    """Load the training configuration schema."""
    from src.utils.schema_loader import resolve_schema
    return resolve_schema(SCHEMA_PATH)


def get_current_language() -> str:
    """Return the active UI language."""
    return st.session_state.get("ui_language", LANG_EN)


def get_ui_text(key: str) -> str:
    """Return localized static UI text."""
    localized_options = UI_STRINGS.get(key, {})
    return localized_options.get(get_current_language()) or localized_options.get(LANG_EN, key)


def get_localized_schema_value(field_schema: Dict[str, Any], key: str, fallback: str = "") -> str:
    """Return localized schema metadata, falling back to English when needed."""
    language = get_current_language()
    localized_key = f"{key}_{language}" if language != LANG_EN else key
    return field_schema.get(localized_key) or field_schema.get(key, fallback)


def get_field_title(field_schema: Dict[str, Any], fallback: str) -> str:
    """Return the localized title for a schema field."""
    return get_localized_schema_value(field_schema, "title", fallback)


def get_field_description(field_schema: Dict[str, Any]) -> str:
    """Return the localized description for a schema field."""
    return get_localized_schema_value(field_schema, "description", "")


def _resolve_default(
    field_schema: Dict[str, Any],
    fallback: Any,
    field_title: str,
) -> Any:
    """Resolve a field's default value from the schema.

    Precedence: ``examples[0]`` → ``default`` → ``minimum`` → ``fallback``.
    Emits a warning when the schema provides none of the first three so the
    missing metadata is surfaced to the developer.
    """
    examples = field_schema.get("examples", [])
    if examples:
        return examples[0]
    if "default" in field_schema:
        return field_schema["default"]
    min_val = field_schema.get("minimum", None)
    if min_val is not None:
        return min_val
    st.warning(
        f"Schema field '{field_title}' has no 'examples', 'default', or "
        f"'minimum'; using fallback value {fallback!r}."
    )
    return fallback


def render_field(
    field_name: str,
    field_schema: Dict[str, Any],
    parent_key: str = "",
    level: int = 0,
) -> Any:
    """Render a form field based on its schema type."""
    # Get title and description from schema
    field_title = get_field_title(field_schema, field_name)
    field_description = get_field_description(field_schema)
    
    # Generate a unique key for the field
    field_key = f"{parent_key}.{field_name}" if parent_key else field_name

    # Handle different types
    field_type = field_schema.get("type")
    
    if field_type == "boolean":
        default = field_schema.get("examples", [False])[0]
        return st.checkbox(field_title, value=default, key=field_key, help=field_description)
    
    elif field_type == "integer":
        min_val = field_schema.get("minimum", None)
        max_val = field_schema.get("maximum", None)
        default = _resolve_default(field_schema, 0, field_title)
        
        return st.number_input(
            field_title,
            value=int(default),
            min_value=int(min_val) if min_val is not None else None,
            max_value=int(max_val) if max_val is not None else None,
            step=1,
            key=field_key,
            help=field_description,
        )
    
    elif field_type == "number":
        min_val = field_schema.get("minimum", None)
        max_val = field_schema.get("maximum", None)
        default = _resolve_default(field_schema, 0.0, field_title)
        
        return st.number_input(
            field_title,
            value=float(default),
            min_value=float(min_val) if min_val is not None else None,
            max_value=float(max_val) if max_val is not None else None,
            step=0.01,
            key=field_key,
            format="%.6f",
            help=field_description,
        )
    
    elif field_type == "string":
        enum = field_schema.get("enum")
        default = _resolve_default(field_schema, "", field_title)
        
        if enum:
            return st.selectbox(field_title, enum, index=0 if enum else 0, key=field_key, help=field_description)
        else:
            return st.text_input(field_title, value=default, key=field_key, help=field_description)
    
    elif field_type == "array":
        items_schema = field_schema.get("items", {})
        enum = items_schema.get("enum")
        examples = field_schema.get("examples", [])
        default = examples[0] if examples else []
        
        if enum:
            st.write(f"**{field_title}**")
            if field_description:
                st.caption(field_description)
            selected = st.multiselect(
                "Select items",
                enum,
                default=default if len(default) > 0 else [enum[0]],
                key=field_key,
            )
            return selected
        else:
            st.write(f"**{field_title}**")
            if field_description:
                st.caption(field_description)
            array_input = st.text_area(
                "Enter items (one per line)",
                value="\n".join(map(str, default)),
                key=field_key,
            )
            return [line.strip() for line in array_input.split("\n") if line.strip()]
    
    elif field_type == "object":
        return render_object(field_name, field_schema, parent_key, level + 1)
    
    return None


def render_object(
    obj_name: str,
    obj_schema: Dict[str, Any],
    parent_key: str = "",
    level: int = 0,
) -> Dict[str, Any]:
    """Render an object schema with all its properties."""
    properties = obj_schema.get("properties", {})
    required = obj_schema.get("required", [])
    
    result = {}
    parent_key = f"{parent_key}.{obj_name}" if parent_key else obj_name
    
    for prop_name, prop_schema in properties.items():
        prop_title = get_field_title(prop_schema, prop_name)
        prop_description = get_field_description(prop_schema)
        is_required = prop_name in required
        expander_label = f"{prop_title} ({get_ui_text('required')})" if is_required else prop_title
        use_expander = level == 0

        if use_expander:
            with st.expander(expander_label, expanded=is_required):
                if prop_description:
                    st.caption(prop_description)

                prop_value = render_field(prop_name, prop_schema, parent_key, level)

                if prop_value is not None:
                    result[prop_name] = prop_value
        else:
            with st.container():
                st.markdown(f"**{expander_label}**")
                if prop_description:
                    st.caption(prop_description)

                prop_value = render_field(prop_name, prop_schema, parent_key, level)

                if prop_value is not None:
                    result[prop_name] = prop_value

                st.divider()
    
    return result


def render_optimizer_section(optimizer_class: str) -> Dict[str, Any]:
    """Render the optimizer configuration section."""
    result: Dict[str, Any] = {
        "optimizer_class": optimizer_class,
    }
    parameters: Dict[str, Any] = {}
    
    # Load schema for optimizer parameters
    schema = load_schema()
    
    with st.expander(get_ui_text("parameter_group"), expanded=False):
        st.info(get_ui_text("parameter_group_info"))
    
    # Map optimizer class to its prefix
    prefix_map = {
        "sgd_momentum": "sgd_momentum",
        "adamw": "adamw",
        "adafactor": "adafactor",
        "galore_adamw": "galore_adamw",
        "prodigy": "prodigy",
        "lion": "lion",
        "sophia": "sophia",
        "muon": "muon",
        "turbo_muon": "turbo_muon",
        "radam": "radam",
        "adan": "adan",
        "adopt": "adopt",
        "ademamix": "ademamix",
        "mars_adamw": "mars_adamw",
        "cautious_adamw": "cautious_adamw",
        "lamb": "lamb",
        "schedulefree_adamw": "schedulefree_adamw",
        "shampoo": "shampoo",
        "soap": "soap",
        "anon": "anon",
        "apollo": "apollo",
        "apollo_mini": "apollo_mini",
        "q_apollo": "q_apollo",
    }
    
    prefix = prefix_map.get(optimizer_class, optimizer_class)
    
    # Descriptions for parameter groups
    param_descriptions = {
        "lr_": "Learning rate controls step size for parameter updates. Higher values converge faster but may overshoot. Lower values are more stable but slower.",
        "wd_": "Weight decay regularizes by penalizing large weights. Higher values prevent overfitting but may underfit.",
        "betas_": "Beta coefficients for momentum (β₁) and squared gradient (β₂). Controls momentum strength and second moment.",
        "eps_": "Epsilon prevents division by zero in adaptive optimizers. Small values ensure numerical stability.",
    }
    
    with st.expander(get_ui_text("embeddings_group"), expanded=False):
        st.caption(get_ui_text("embeddings_caption"))
        lr_emb = st.number_input(
            "Learning Rate",
            value=1e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_embeddings",
            help=f"{prefix}-lr_embeddings: {param_descriptions['lr_']}",
        )
        wd_emb = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_embeddings",
            help=f"{prefix}-wd_embeddings: {param_descriptions['wd_']}",
        )
        parameters[f"{prefix}-lr_embeddings"] = lr_emb
        parameters[f"{prefix}-wd_embeddings"] = wd_emb
    with st.expander(get_ui_text("normalization_group"), expanded=False):
        st.caption(get_ui_text("normalization_caption"))
        lr_norm = st.number_input(
            "Learning Rate",
            value=5e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_norms",
            help=f"{prefix}-lr_norms: {param_descriptions['lr_']}",
        )
        wd_norm = st.number_input(
            "Weight Decay",
            value=0.001,
            min_value=0.0,
            format="%.4f",
            key=f"{prefix}-wd_norms",
            help=f"{prefix}-wd_norms: {param_descriptions['wd_']}",
        )
        parameters[f"{prefix}-lr_norms"] = lr_norm
        parameters[f"{prefix}-wd_norms"] = wd_norm
    
    with st.expander(get_ui_text("attention_group"), expanded=False):
        st.caption(get_ui_text("attention_caption"))
        lr_attn = st.number_input(
            "Learning Rate",
            value=3e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_attention",
            help=f"{prefix}-lr_attention: {param_descriptions['lr_']}",
        )
        wd_attn = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_attention",
            help=f"{prefix}-wd_attention: {param_descriptions['wd_']}",
        )
        parameters[f"{prefix}-lr_attention"] = lr_attn
        parameters[f"{prefix}-wd_attention"] = wd_attn
    
    with st.expander(get_ui_text("other_group"), expanded=False):
        st.caption(get_ui_text("other_caption"))
        lr_other = st.number_input(
            "Learning Rate",
            value=2e-6,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-lr_other",
            help=f"{prefix}-lr_other: {param_descriptions['lr_']}",
        )
        wd_other = st.number_input(
            "Weight Decay",
            value=0.01,
            min_value=0.0,
            format="%.3f",
            key=f"{prefix}-wd_other",
            help=f"{prefix}-wd_other: {param_descriptions['wd_']}",
        )
        betas_other = st.text_input(
            "Betas",
            value="[0.9, 0.95]",
            key=f"{prefix}-betas_other",
            help=f"{prefix}-betas_other: {param_descriptions['betas_']}",
        )
        eps_other = st.number_input(
            "Epsilon",
            value=1e-8,
            min_value=0.0,
            format="%.1e",
            key=f"{prefix}-eps_other",
            help=f"{prefix}-eps_other: {param_descriptions['eps_']}",
        )
        parameters[f"{prefix}-lr_other"] = lr_other
        parameters[f"{prefix}-wd_other"] = wd_other
        parameters[f"{prefix}-betas_other"] = betas_other
        parameters[f"{prefix}-eps_other"] = eps_other

    result["parameters"] = parameters
    return result


def render_sbert_section(training_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Render the ``training.sbert`` configuration block.

    Shown only when ``training.task == "sbert"``. Reads the SBERT subsection
    from the training schema and produces a nested ``sbert`` dict.

    Args:
        training_schema: The resolved ``training`` property schema.

    Returns:
        The SBERT configuration dict, or an empty dict if the schema
        does not expose an ``sbert`` property.
    """
    sbert_props = (
        training_schema.get("properties", {}).get("sbert", {}).get("properties", {})
    )
    if not sbert_props:
        return {}

    def field_title(name, fallback=""):
        return get_field_title(sbert_props.get(name, {}), fallback)

    def field_help(name):
        return get_field_description(sbert_props.get(name, {}))

    def field_example(name, default):
        examples = sbert_props.get(name, {}).get("examples", [])
        return examples[0] if examples else default

    result: Dict[str, Any] = {}

    with st.expander("SBERT Configuration", expanded=True):
        result["dataset_name"] = st.text_input(
            field_title("dataset_name", "Dataset Name"),
            value=str(field_example("dataset_name", "erickfmm/agentlans__multilingual-sentences__paired_10_sts")),
            key="sbert.dataset_name",
            help=field_help("dataset_name"),
        )
        dataset_type_schema = sbert_props.get("dataset_type", {})
        result["dataset_type"] = st.selectbox(
            field_title("dataset_type", "Dataset Type"),
            dataset_type_schema.get("enum", ["paired_similarity", "triplets", "qa"]),
            index=0,
            key="sbert.dataset_type",
            help=field_help("dataset_type"),
        )
        result["output_dir"] = st.text_input(
            field_title("output_dir", "Output Dir"),
            value=str(field_example("output_dir", "./output/sbert_model")),
            key="sbert.output_dir",
            help=field_help("output_dir"),
        )
        result["batch_size"] = st.number_input(
            field_title("batch_size", "Batch Size"),
            value=int(field_example("batch_size", 16)),
            min_value=1,
            key="sbert.batch_size",
            help=field_help("batch_size"),
        )
        result["gradient_accumulation_steps"] = st.number_input(
            field_title("gradient_accumulation_steps", "Gradient Accumulation Steps"),
            value=int(field_example("gradient_accumulation_steps", 1)),
            min_value=1,
            key="sbert.gradient_accumulation_steps",
            help=field_help("gradient_accumulation_steps"),
        )
        result["max_grad_norm"] = st.number_input(
            field_title("max_grad_norm", "Max Gradient Norm"),
            value=float(field_example("max_grad_norm", 1.0)),
            min_value=0.0,
            key="sbert.max_grad_norm",
            help=field_help("max_grad_norm"),
        )
        result["epochs"] = st.number_input(
            field_title("epochs", "Epochs"),
            value=int(field_example("epochs", 4)),
            min_value=1,
            key="sbert.epochs",
            help=field_help("epochs"),
        )
        result["warmup_steps"] = st.number_input(
            field_title("warmup_steps", "Warmup Steps"),
            value=int(field_example("warmup_steps", 1000)),
            min_value=0,
            key="sbert.warmup_steps",
            help=field_help("warmup_steps"),
        )
        result["evaluation_steps"] = st.number_input(
            field_title("evaluation_steps", "Evaluation Steps"),
            value=int(field_example("evaluation_steps", 5000)),
            min_value=1,
            key="sbert.evaluation_steps",
            help=field_help("evaluation_steps"),
        )
        result["checkpoint_save_steps"] = st.number_input(
            field_title("checkpoint_save_steps", "Checkpoint Save Interval"),
            value=int(field_example("checkpoint_save_steps", 1000)),
            min_value=0,
            key="sbert.checkpoint_save_steps",
            help=field_help("checkpoint_save_steps"),
        )
        result["learning_rate"] = st.number_input(
            field_title("learning_rate", "Learning Rate"),
            value=float(field_example("learning_rate", 2e-5)),
            min_value=0.0,
            format="%.2e",
            key="sbert.learning_rate",
            help=field_help("learning_rate"),
        )
        result["max_train_samples"] = st.number_input(
            field_title("max_train_samples", "Max Training Samples"),
            value=int(field_example("max_train_samples", 100000)),
            min_value=1,
            key="sbert.max_train_samples",
            help=field_help("max_train_samples"),
        )
        result["max_eval_samples"] = st.number_input(
            field_title("max_eval_samples", "Max Eval Samples"),
            value=int(field_example("max_eval_samples", 10000)),
            min_value=1,
            key="sbert.max_eval_samples",
            help=field_help("max_eval_samples"),
        )
        result["max_seq_length"] = st.number_input(
            field_title("max_seq_length", "Max Sequence Length"),
            value=int(field_example("max_seq_length", 512)),
            min_value=1,
            key="sbert.max_seq_length",
            help=field_help("max_seq_length"),
        )
        pooling_schema = sbert_props.get("pooling_mode", {})
        result["pooling_mode"] = st.selectbox(
            field_title("pooling_mode", "Pooling Mode"),
            pooling_schema.get("enum", ["mean", "cls", "max"]),
            index=0,
            key="sbert.pooling_mode",
            help=field_help("pooling_mode"),
        )
        result["resample_balanced"] = st.checkbox(
            field_title("resample_balanced", "Resample Balanced"),
            value=bool(field_example("resample_balanced", False)),
            key="sbert.resample_balanced",
            help=field_help("resample_balanced"),
        )
        result["standardize_scores"] = st.checkbox(
            field_title("standardize_scores", "Standardize Scores"),
            value=bool(field_example("standardize_scores", False)),
            key="sbert.standardize_scores",
            help=field_help("standardize_scores"),
        )
        result["resample_std"] = st.number_input(
            field_title("resample_std", "Resample Std Dev"),
            value=float(field_example("resample_std", 0.3)),
            min_value=0.0,
            key="sbert.resample_std",
            help=field_help("resample_std"),
        )
        result["trust_remote_code"] = st.checkbox(
            field_title("trust_remote_code", "Trust Remote Code"),
            value=bool(field_example("trust_remote_code", False)),
            key="sbert.trust_remote_code",
            help=field_help("trust_remote_code"),
        )
        result["use_amp"] = st.checkbox(
            field_title("use_amp", "Use AMP"),
            value=bool(field_example("use_amp", True)),
            key="sbert.use_amp",
            help=field_help("use_amp"),
        )

    return result


def build_config_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Build the configuration dictionary from the schema and user input."""
    config = {}
    
    # Model configuration
    st.header("Model Configuration")
    st.subheader("Choose Model Type")
    model_mode = st.radio(
        "Model Mode",
        ["Custom Model", "Base Model"],
        horizontal=True,
    )
    
    if model_mode == "Custom Model":
        with st.expander("Model Parameters", expanded=True):
            model_schema = schema["properties"]["model"]
            model_description = get_field_description(model_schema)
            if model_description:
                st.caption(model_description)
            model_config = render_object("model", schema["properties"]["model"])
            config["model_class"] = st.selectbox(
                get_field_title(schema["properties"]["model_class"], "Model Class"),
                schema["properties"]["model_class"]["enum"],
                index=0,
                key="model_class",
                help=get_field_description(schema["properties"]["model_class"]),
            )
            config["model"] = model_config
    else:
        with st.expander("Base Model Configuration", expanded=True):
            base_model = st.text_input(
                get_field_title(schema["properties"]["base_model"], "Base Model ID/Path"),
                value="answerdotai/ModernBERT-base",
                key="base_model",
                help=get_field_description(schema["properties"]["base_model"]),
            )
            config["base_model"] = base_model
            
            tokenizer_schema = schema["properties"]["tokenizer"]
            st.write(f"**{get_field_title(tokenizer_schema, 'Tokenizer Configuration')}**")
            tokenizer_description = get_field_description(tokenizer_schema)
            if tokenizer_description:
                st.caption(tokenizer_description)
            tokenizer_config = render_object("tokenizer", schema["properties"]["tokenizer"])
            config["tokenizer"] = tokenizer_config
    
    # Training configuration
    st.header("Training Configuration")
    with st.expander("Training Parameters", expanded=True):
        training_schema = schema["properties"]["training"]
        
        # Task selection
        task_schema = training_schema["properties"]["task"]
        task = st.selectbox(
            get_field_title(task_schema, "Training Task"),
            task_schema["enum"],
            index=0,
            key="training.task",
            help=get_field_description(task_schema),
        )
        
        training_config = {"task": task}

        if task == "sbert":
            sbert_config = render_sbert_section(training_schema)
            if sbert_config:
                training_config["sbert"] = sbert_config
        
        # General training parameters
        with st.expander("General Training Parameters", expanded=True):
            num_epochs_schema = training_schema["properties"]["num_epochs"]
            batch_size_schema = training_schema["properties"]["batch_size"]
            max_length_schema = training_schema["properties"]["max_length"]

            num_epochs = st.number_input(
                get_field_title(num_epochs_schema, "Number of Epochs"),
                value=5,
                min_value=1,
                key="training.num_epochs",
                help=get_field_description(num_epochs_schema),
            )
            batch_size = st.number_input(
                get_field_title(batch_size_schema, "Batch Size"),
                value=4,
                min_value=1,
                key="training.batch_size",
                help=get_field_description(batch_size_schema),
            )
            max_length = st.number_input(
                get_field_title(max_length_schema, "Max Sequence Length"),
                value=512,
                min_value=1,
                key="training.max_length",
                help=get_field_description(max_length_schema),
            )

            training_config["num_epochs"] = num_epochs
            training_config["batch_size"] = batch_size
            training_config["max_length"] = max_length
        
        # Dataset parameters
        max_samples_schema = training_schema["properties"]["max_samples"]
        dataset_batch_size_schema = training_schema["properties"]["dataset_batch_size"]
        with st.expander("Dataset Parameters", expanded=False):
            max_samples = st.number_input(
                get_field_title(max_samples_schema, "Max Samples"),
                value=20000000,
                min_value=1,
                key="training.max_samples",
                help=get_field_description(max_samples_schema),
            )
            dataset_batch_size = st.number_input(
                get_field_title(dataset_batch_size_schema, "Dataset Batch Size"),
                value=25000,
                min_value=1,
                key="training.dataset_batch_size",
                help=get_field_description(dataset_batch_size_schema),
            )

            training_config["max_samples"] = max_samples
            training_config["dataset_batch_size"] = dataset_batch_size
        
        # Optimizer
        optimizer_class_schema = schema["properties"]["training"]["properties"]["optimizer"]["properties"]["optimizer_class"]
        with st.expander("Optimizer", expanded=False):
            optimizer_class = st.selectbox(
                get_field_title(optimizer_class_schema, "Optimizer Class"),
                optimizer_class_schema["enum"],
                index=1,  # Default to adamw
                key="training.optimizer.optimizer_class",
                help=get_field_description(optimizer_class_schema),
            )

            optimizer_params = render_optimizer_section(optimizer_class)
        training_config["optimizer"] = optimizer_params
        
        # Scheduler
        scheduler_total_steps_schema = training_schema["properties"]["scheduler_total_steps"]
        scheduler_warmup_ratio_schema = training_schema["properties"]["scheduler_warmup_ratio"]
        scheduler_type_schema = training_schema["properties"]["scheduler_type"]
        with st.expander("Scheduler", expanded=False):
            scheduler_total_steps = st.number_input(
                get_field_title(scheduler_total_steps_schema, "Total Steps"),
                value=10000,
                min_value=1,
                key="training.scheduler_total_steps",
                help=get_field_description(scheduler_total_steps_schema),
            )
            scheduler_warmup_ratio = st.number_input(
                get_field_title(scheduler_warmup_ratio_schema, "Warmup Ratio"),
                value=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="training.scheduler_warmup_ratio",
                help=get_field_description(scheduler_warmup_ratio_schema),
            )
            scheduler_type = st.selectbox(
                get_field_title(scheduler_type_schema, "Scheduler Type"),
                scheduler_type_schema.get("enum", []),
                index=0,
                key="training.scheduler_type",
                help=get_field_description(scheduler_type_schema),
            )

            training_config["scheduler_total_steps"] = scheduler_total_steps
            training_config["scheduler_warmup_ratio"] = scheduler_warmup_ratio
            training_config["scheduler_type"] = scheduler_type
        
        # Gradient settings
        gradient_accumulation_steps_schema = training_schema["properties"]["gradient_accumulation_steps"]
        grad_clip_max_norm_schema = training_schema["properties"]["grad_clip_max_norm"]
        with st.expander("Gradient Settings", expanded=False):
            gradient_accumulation_steps = st.number_input(
                get_field_title(gradient_accumulation_steps_schema, "Gradient Accumulation Steps"),
                value=4,
                min_value=1,
                key="training.gradient_accumulation_steps",
                help=get_field_description(gradient_accumulation_steps_schema),
            )
            grad_clip_max_norm = st.number_input(
                get_field_title(grad_clip_max_norm_schema, "Gradient Clip Max Norm"),
                value=5.0,
                min_value=0.0,
                key="training.grad_clip_max_norm",
                help=get_field_description(grad_clip_max_norm_schema),
            )

            training_config["gradient_accumulation_steps"] = gradient_accumulation_steps
            training_config["grad_clip_max_norm"] = grad_clip_max_norm
        
        # Checkpoint settings
        checkpoint_every_n_steps_schema = training_schema["properties"]["checkpoint_every_n_steps"]
        max_rolling_checkpoints_schema = training_schema["properties"]["max_rolling_checkpoints"]
        num_best_checkpoints_schema = training_schema["properties"]["num_best_checkpoints"]
        with st.expander("Checkpoint Settings", expanded=False):
            checkpoint_every_n_steps = st.number_input(
                get_field_title(checkpoint_every_n_steps_schema, "Checkpoint Every N Steps"),
                value=500,
                min_value=1,
                key="training.checkpoint_every_n_steps",
                help=get_field_description(checkpoint_every_n_steps_schema),
            )
            max_rolling_checkpoints = st.number_input(
                get_field_title(max_rolling_checkpoints_schema, "Max Rolling Checkpoints"),
                value=3,
                min_value=1,
                key="training.max_rolling_checkpoints",
                help=get_field_description(max_rolling_checkpoints_schema),
            )
            num_best_checkpoints = st.number_input(
                get_field_title(num_best_checkpoints_schema, "Num Best Checkpoints"),
                value=2,
                min_value=1,
                key="training.num_best_checkpoints",
                help=get_field_description(num_best_checkpoints_schema),
            )

            training_config["checkpoint_every_n_steps"] = checkpoint_every_n_steps
            training_config["max_rolling_checkpoints"] = max_rolling_checkpoints
            training_config["num_best_checkpoints"] = num_best_checkpoints
        
        # Logging settings
        csv_log_path_schema = training_schema["properties"]["csv_log_path"]
        log_gradient_stats_schema = training_schema["properties"]["log_gradient_stats"]
        gradient_log_interval_schema = training_schema["properties"]["gradient_log_interval"]
        with st.expander("Logging Settings", expanded=False):
            csv_log_path = st.text_input(
                get_field_title(csv_log_path_schema, "CSV Log Path"),
                value="training_metrics.csv",
                key="training.csv_log_path",
                help=get_field_description(csv_log_path_schema),
            )
            log_gradient_stats = st.checkbox(
                get_field_title(log_gradient_stats_schema, "Log Gradient Stats"),
                value=True,
                key="training.log_gradient_stats",
                help=get_field_description(log_gradient_stats_schema),
            )
            gradient_log_interval = st.number_input(
                get_field_title(gradient_log_interval_schema, "Gradient Log Interval"),
                value=10,
                min_value=1,
                key="training.gradient_log_interval",
                help=get_field_description(gradient_log_interval_schema),
            )

            training_config["csv_log_path"] = csv_log_path
            training_config["log_gradient_stats"] = log_gradient_stats
            training_config["gradient_log_interval"] = gradient_log_interval
        
        # GPU settings
        gpu_temp_guard_enabled_schema = training_schema["properties"]["gpu_temp_guard_enabled"]
        gpu_temp_pause_threshold_c_schema = training_schema["properties"]["gpu_temp_pause_threshold_c"]
        gpu_temp_resume_threshold_c_schema = training_schema["properties"]["gpu_temp_resume_threshold_c"]
        gpu_temp_critical_threshold_c_schema = training_schema["properties"]["gpu_temp_critical_threshold_c"]
        gpu_temp_poll_interval_schema = training_schema["properties"]["gpu_temp_poll_interval_seconds"]
        gpu_temp_grace_schema = training_schema["properties"]["gpu_temp_checkpoint_grace_seconds"]
        switch_on_thermal_schema = training_schema["properties"]["switch_on_thermal"]
        resume_from_checkpoint_schema = training_schema["properties"]["resume_from_checkpoint"]
        with st.expander("GPU Settings", expanded=False):
            gpu_temp_guard_enabled = st.checkbox(
                get_field_title(gpu_temp_guard_enabled_schema, "Enable GPU Temperature Guard"),
                value=True,
                key="training.gpu_temp_guard_enabled",
                help=get_field_description(gpu_temp_guard_enabled_schema),
            )
            gpu_temp_pause_threshold_c = st.number_input(
                get_field_title(gpu_temp_pause_threshold_c_schema, "GPU Temp Pause Threshold (°C)"),
                value=90.0,
                min_value=0.0,
                key="training.gpu_temp_pause_threshold_c",
                help=get_field_description(gpu_temp_pause_threshold_c_schema),
            )
            gpu_temp_resume_threshold_c = st.number_input(
                get_field_title(gpu_temp_resume_threshold_c_schema, "GPU Temp Resume Threshold (°C)"),
                value=80.0,
                min_value=0.0,
                key="training.gpu_temp_resume_threshold_c",
                help=get_field_description(gpu_temp_resume_threshold_c_schema),
            )
            gpu_temp_critical_threshold_c = st.number_input(
                get_field_title(gpu_temp_critical_threshold_c_schema, "GPU Temp Critical Threshold (°C)"),
                value=0.0,
                min_value=0.0,
                key="training.gpu_temp_critical_threshold_c",
                help=get_field_description(gpu_temp_critical_threshold_c_schema),
            )
            training_config["gpu_temp_critical_threshold_c"] = (
                gpu_temp_critical_threshold_c if gpu_temp_critical_threshold_c > 0 else None
            )
            gpu_temp_poll_interval_seconds = st.number_input(
                get_field_title(gpu_temp_poll_interval_schema, "GPU Temp Poll Interval (s)"),
                value=30.0,
                min_value=0.0,
                key="training.gpu_temp_poll_interval_seconds",
                help=get_field_description(gpu_temp_poll_interval_schema),
            )
            training_config["gpu_temp_poll_interval_seconds"] = gpu_temp_poll_interval_seconds
            gpu_temp_checkpoint_grace_seconds = st.number_input(
                get_field_title(gpu_temp_grace_schema, "GPU Temp Checkpoint Grace (s)"),
                value=30.0,
                min_value=0.0,
                key="training.gpu_temp_checkpoint_grace_seconds",
                help=get_field_description(gpu_temp_grace_schema),
            )
            training_config["gpu_temp_checkpoint_grace_seconds"] = gpu_temp_checkpoint_grace_seconds
            switch_on_thermal = st.checkbox(
                get_field_title(switch_on_thermal_schema, "Switch on Thermal (continue on CPU at critical temp)"),
                value=False,
                key="training.switch_on_thermal",
                help=get_field_description(switch_on_thermal_schema),
            )
            training_config["switch_on_thermal"] = switch_on_thermal
            resume_from_checkpoint = st.text_input(
                get_field_title(resume_from_checkpoint_schema, "Resume From Checkpoint (auto or path)"),
                value="",
                key="training.resume_from_checkpoint",
                help=get_field_description(resume_from_checkpoint_schema),
            )
            training_config["resume_from_checkpoint"] = resume_from_checkpoint.strip() or None

            training_config["gpu_temp_guard_enabled"] = gpu_temp_guard_enabled
            training_config["gpu_temp_pause_threshold_c"] = gpu_temp_pause_threshold_c
            training_config["gpu_temp_resume_threshold_c"] = gpu_temp_resume_threshold_c
        
        # Mixed precision
        use_amp_schema = training_schema["properties"]["use_amp"]
        use_amp = st.checkbox(
            get_field_title(use_amp_schema, "Use Automatic Mixed Precision"),
            value=False,
            key="training.use_amp",
            help=get_field_description(use_amp_schema),
        )
        training_config["use_amp"] = use_amp
        
        config["training"] = training_config

    # Causal-LM requires the decoder model class. Enforce at config build time.
    if config.get("training", {}).get("task") == "causal_lm":
        if "base_model" not in config:
            config["model_class"] = "frankensteindecoder"
            st.warning(
                "Training task 'causal_lm' requires the decoder model class. "
                "'model_class' has been set to 'frankensteindecoder'."
            )

    return config


def _add_optional(
    cmd_parts: List[str], flag: str, value: Any, *, store_true: bool = False
) -> None:
    """Append an optional CLI flag to ``cmd_parts`` when ``value`` is truthy.

    For ``store_true`` flags only the flag is appended (no value). ``value``
    may be a boolean (enables/disables the flag) or a scalar that is skipped
    when empty/None.
    """
    if store_true:
        if value:
            cmd_parts.append(flag)
        return
    if value is None or value == "" or value == []:
        return
    if isinstance(value, bool):
        cmd_parts.append(flag)
        return
    cmd_parts.extend([flag, str(value)])


def build_cli_command(
    command: str,
    config: Dict[str, Any],
    output_path: str,
    extra_args: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the CLI command based on the selected command and configuration.

    Each subcommand builds the argument list expected by ``src/cli.py``.
    ``extra_args`` carries command-specific inputs collected from the UI
    (e.g. ``--checkpoint``/``--output`` for deploy, ``--model`` for infer).

    Args:
        command: One of the supported CLI subcommands.
        config: The (training) configuration dict built by the builder.
        output_path: Path of the generated YAML config file.
        extra_args: Optional mapping of command-specific argument values.

    Returns:
        The fully-formed CLI command as a string.
    """
    extra_args = extra_args or {}
    cmd_parts = ["python", "-m", "src.cli", command]

    def add(flag, value, *, store_true=False):
        _add_optional(cmd_parts, flag, value, store_true=store_true)

    if command == "train":
        add("--config", output_path)
        add("--config-name", config.get("model_class", "frankenstein"))
        add("--device", extra_args.get("device"))
        add("--batch-size", extra_args.get("batch_size"))
        add("--gpu-temp-guard", extra_args.get("gpu_temp_guard"), store_true=True)
        add("--no-gpu-temp-guard", extra_args.get("gpu_temp_guard") is False, store_true=True)
        add("--gpu-temp-pause-threshold-c", extra_args.get("gpu_temp_pause_threshold_c"))
        add("--gpu-temp-resume-threshold-c", extra_args.get("gpu_temp_resume_threshold_c"))
        add("--gpu-temp-critical-threshold-c", extra_args.get("gpu_temp_critical_threshold_c"))
        add("--gpu-temp-poll-interval-seconds", extra_args.get("gpu_temp_poll_interval_seconds"))
        add("--gpu-temp-checkpoint-grace-seconds", extra_args.get("gpu_temp_checkpoint_grace_seconds"))
        add("--resume-from-checkpoint", extra_args.get("resume_from_checkpoint"))
        add("--switch-on-thermal", extra_args.get("switch_on_thermal"), store_true=True)
        add("--no-switch-on-thermal", extra_args.get("switch_on_thermal") is False, store_true=True)
    elif command in ["deploy", "quantize"]:
        add("--checkpoint", extra_args.get("checkpoint"))
        add("--output", extra_args.get("output"))
        add("--format", extra_args.get("format"))
        add("--device", extra_args.get("device"))
        add("--config", extra_args.get("config"))
        add("--yaml", extra_args.get("yaml"))
        add("--validate", extra_args.get("validate"), store_true=True)
    elif command == "infer":
        add("--model", extra_args.get("model"))
        add("--device", extra_args.get("device"))
        add("--batch-size", extra_args.get("batch_size"))
        add("--text", extra_args.get("text"))
        add("--input", extra_args.get("input"))
        add("--output", extra_args.get("output"))
        add("--fp16", extra_args.get("fp16"), store_true=True)
        add("--benchmark", extra_args.get("benchmark"), store_true=True)
    elif command == "sbert-train":
        add("--base-model", extra_args.get("base_model"))
        add("--pretrained", extra_args.get("pretrained"))
        add("--output_dir", extra_args.get("output_dir"))
        add("--dataset_name", extra_args.get("dataset_name"))
        add("--batch_size", extra_args.get("batch_size"))
        add("--epochs", extra_args.get("epochs"))
        add("--warmup_steps", extra_args.get("warmup_steps"))
        add("--evaluation_steps", extra_args.get("evaluation_steps"))
        add("--learning_rate", extra_args.get("learning_rate"))
        add("--max_train_samples", extra_args.get("max_train_samples"))
        add("--max_eval_samples", extra_args.get("max_eval_samples"))
        add("--max_seq_length", extra_args.get("max_seq_length"))
        add("--hidden_size", extra_args.get("hidden_size"))
        add("--num_layers", extra_args.get("num_layers"))
        add("--pooling_mode", extra_args.get("pooling_mode"))
        add("--resample_std", extra_args.get("resample_std"))
        add("--device", extra_args.get("device"))
        add("--trust_remote_code", extra_args.get("trust_remote_code"), store_true=True)
        add("--no_amp", extra_args.get("no_amp"), store_true=True)
        add("--no_resample", extra_args.get("no_resample"), store_true=True)
        add("--switch-on-thermal", extra_args.get("switch_on_thermal"), store_true=True)
        add("--no-switch-on-thermal", extra_args.get("switch_on_thermal") is False, store_true=True)
    elif command == "sbert-infer":
        add("--model_path", extra_args.get("model_path"))
        add("--mode", extra_args.get("mode"))
        add("--batch_size", extra_args.get("batch_size"))
        add("--device", extra_args.get("device"))
        add("--top_k", extra_args.get("top_k"))
        add("--n_clusters", extra_args.get("n_clusters"))
        add("--sentence1", extra_args.get("sentence1"))
        add("--sentence2", extra_args.get("sentence2"))
        add("--query", extra_args.get("query"))
        add("--corpus_file", extra_args.get("corpus_file"))
        add("--sentences_file", extra_args.get("sentences_file"))
        add("--input_file", extra_args.get("input_file"))
        add("--output_file", extra_args.get("output_file"))

    return " ".join(str(part) for part in cmd_parts)


def render_command_args(command: str) -> Dict[str, Any]:
    """Render command-specific argument inputs for the selected subcommand.

    Returns a dict of argument values that is passed to
    :func:`build_cli_command` as ``extra_args``. Keys match the flags the
    CLI expects for each subcommand.

    Args:
        command: The selected CLI subcommand.

    Returns:
        Mapping of argument key -> value collected from the UI.
    """
    args: Dict[str, Any] = {}

    device = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        index=0,
        key=f"{command}.device",
    )

    if command == "train":
        args["device"] = device
        args["batch_size"] = st.number_input(
            "Batch Size (override)", value=0, min_value=0, step=1, key="train.batch_size"
        ) or None
        args["gpu_temp_guard"] = st.checkbox(
            "Enable GPU Temp Guard", value=True, key="train.gpu_temp_guard"
        )
        args["switch_on_thermal"] = st.checkbox(
            "Switch on Thermal", value=False, key="train.switch_on_thermal"
        )
        args["resume_from_checkpoint"] = st.text_input(
            "Resume From Checkpoint (auto or path)", value="", key="train.resume"
        ).strip() or None
        with st.expander("Advanced GPU Temp Thresholds", expanded=False):
            args["gpu_temp_pause_threshold_c"] = st.number_input(
                "GPU Temp Pause (°C)", value=0.0, min_value=0.0, key="train.temp_pause"
            ) or None
            args["gpu_temp_resume_threshold_c"] = st.number_input(
                "GPU Temp Resume (°C)", value=0.0, min_value=0.0, key="train.temp_resume"
            ) or None
            args["gpu_temp_critical_threshold_c"] = st.number_input(
                "GPU Temp Critical (°C)", value=0.0, min_value=0.0, key="train.temp_critical"
            ) or None
            args["gpu_temp_poll_interval_seconds"] = st.number_input(
                "GPU Temp Poll Interval (s)", value=0.0, min_value=0.0, key="train.temp_poll"
            ) or None
            args["gpu_temp_checkpoint_grace_seconds"] = st.number_input(
                "GPU Temp Checkpoint Grace (s)", value=0.0, min_value=0.0, key="train.temp_grace"
            ) or None
    elif command in ["deploy", "quantize"]:
        args["device"] = device
        args["checkpoint"] = st.text_input(
            "Checkpoint Path", value="", key=f"{command}.checkpoint"
        ).strip() or None
        args["output"] = st.text_input(
            "Output Directory", value="", key=f"{command}.output"
        ).strip() or None
        if command == "deploy":
            args["format"] = st.selectbox(
                "Deploy Format",
                ["quantized", "standard"],
                index=0,
                key="deploy.format",
            )
        args["validate"] = st.checkbox(
            "Validate Output", value=False, key=f"{command}.validate"
        )
        args["config"] = st.text_input(
            "Training Config YAML (optional)", value="", key=f"{command}.config"
        ).strip() or None
        args["yaml"] = st.text_input(
            "Model YAML (for transformers-export, optional)", value="", key=f"{command}.yaml"
        ).strip() or None
    elif command == "infer":
        args["device"] = device
        args["model"] = st.text_input(
            "Model Path", value="", key="infer.model"
        ).strip() or None
        args["batch_size"] = st.number_input(
            "Batch Size", value=8, min_value=1, step=1, key="infer.batch_size"
        )
        args["text"] = st.text_input(
            "Text (single prompt)", value="", key="infer.text"
        ).strip() or None
        args["input"] = st.text_input(
            "Input File", value="", key="infer.input"
        ).strip() or None
        args["output"] = st.text_input(
            "Output File", value="", key="infer.output"
        ).strip() or None
        args["fp16"] = st.checkbox("Use FP16", value=False, key="infer.fp16")
        args["benchmark"] = st.checkbox("Run Benchmark", value=False, key="infer.benchmark")
    elif command == "sbert-train":
        args["device"] = device
        args["base_model"] = st.text_input(
            "Base Model", value="", key="sbert_train.base_model"
        ).strip() or None
        args["pretrained"] = st.text_input(
            "Pretrained Model", value="", key="sbert_train.pretrained"
        ).strip() or None
        args["output_dir"] = st.text_input(
            "Output Dir", value="./output/sbert_frankenstein_v2", key="sbert_train.output_dir"
        ).strip()
        args["dataset_name"] = st.text_input(
            "Dataset Name",
            value="erickfmm/agentlans__multilingual-sentences__paired_10_sts",
            key="sbert_train.dataset_name",
        ).strip()
        args["batch_size"] = st.number_input(
            "Batch Size", value=16, min_value=1, step=1, key="sbert_train.batch_size"
        )
        args["epochs"] = st.number_input(
            "Epochs", value=4, min_value=1, step=1, key="sbert_train.epochs"
        )
        args["warmup_steps"] = st.number_input(
            "Warmup Steps", value=1000, min_value=0, step=1, key="sbert_train.warmup_steps"
        )
        args["evaluation_steps"] = st.number_input(
            "Evaluation Steps", value=5000, min_value=1, step=1, key="sbert_train.eval_steps"
        )
        args["learning_rate"] = st.number_input(
            "Learning Rate", value=2e-5, min_value=0.0, format="%.2e", key="sbert_train.lr"
        )
        args["max_seq_length"] = st.number_input(
            "Max Sequence Length", value=512, min_value=1, step=1, key="sbert_train.max_len"
        )
        args["hidden_size"] = st.number_input(
            "Hidden Size", value=768, min_value=1, step=1, key="sbert_train.hidden"
        )
        args["num_layers"] = st.number_input(
            "Num Layers", value=12, min_value=1, step=1, key="sbert_train.layers"
        )
        args["pooling_mode"] = st.selectbox(
            "Pooling Mode", ["mean", "cls", "max"], index=0, key="sbert_train.pooling"
        )
        args["resample_std"] = st.number_input(
            "Resample Std", value=0.3, min_value=0.0, key="sbert_train.resample_std"
        )
        args["trust_remote_code"] = st.checkbox(
            "Trust Remote Code", value=False, key="sbert_train.trust_remote"
        )
        args["no_amp"] = st.checkbox(
            "Disable AMP", value=False, key="sbert_train.no_amp"
        )
        args["no_resample"] = st.checkbox(
            "Disable Resample", value=False, key="sbert_train.no_resample"
        )
        args["switch_on_thermal"] = st.checkbox(
            "Switch on Thermal", value=False, key="sbert_train.switch_on_thermal"
        )
    elif command == "sbert-infer":
        args["device"] = device
        args["model_path"] = st.text_input(
            "Model Path", value="", key="sbert_infer.model_path"
        ).strip() or None
        args["mode"] = st.selectbox(
            "Inference Mode",
            ["similarity", "search", "cluster", "encode"],
            index=0,
            key="sbert_infer.mode",
        )
        args["batch_size"] = st.number_input(
            "Batch Size", value=32, min_value=1, step=1, key="sbert_infer.batch_size"
        )
        args["top_k"] = st.number_input(
            "Top K", value=5, min_value=1, step=1, key="sbert_infer.top_k"
        )
        args["n_clusters"] = st.number_input(
            "N Clusters", value=5, min_value=1, step=1, key="sbert_infer.n_clusters"
        )
        args["sentence1"] = st.text_input(
            "Sentence 1", value="", key="sbert_infer.sentence1"
        ).strip() or None
        args["sentence2"] = st.text_input(
            "Sentence 2", value="", key="sbert_infer.sentence2"
        ).strip() or None
        args["query"] = st.text_input(
            "Query", value="", key="sbert_infer.query"
        ).strip() or None
        args["corpus_file"] = st.text_input(
            "Corpus File", value="", key="sbert_infer.corpus_file"
        ).strip() or None
        args["sentences_file"] = st.text_input(
            "Sentences File", value="", key="sbert_infer.sentences_file"
        ).strip() or None
        args["input_file"] = st.text_input(
            "Input File", value="", key="sbert_infer.input_file"
        ).strip() or None
        args["output_file"] = st.text_input(
            "Output File", value="", key="sbert_infer.output_file"
        ).strip() or None

    return args


def run_command_with_nohup(command: str, log_file: str) -> subprocess.Popen:
    """Run a command with nohup in the background."""
    nohup_cmd = f"nohup {command} > {log_file} 2>&1 &"
    process = subprocess.Popen(
        nohup_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process


def main(argv=None):
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="Transformer Encoder Frankenstein - Config Builder",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Transformer Encoder Frankenstein - Config Builder")
    st.markdown("Build YAML configuration files and generate CLI commands dynamically.")
    
    # Load schema
    try:
        schema = load_schema()
        st.success("Schema loaded successfully!")
    except Exception as e:
        st.error(f"Error loading schema: {e}")
        return
    
    # Sidebar for command selection
    if "ui_language" not in st.session_state:
        st.session_state["ui_language"] = LANG_EN

    language_options = list(LANGUAGE_LABELS.keys())
    st.sidebar.selectbox(
        get_ui_text("language_selector"),
        language_options,
        index=language_options.index(get_current_language()),
        key="ui_language",
        format_func=lambda language_code: LANGUAGE_LABELS[language_code],
    )

    st.sidebar.header("Command Selection")
    command_info = st.sidebar.selectbox(
        "Select Command",
        AVAILABLE_COMMANDS,
        format_func=lambda x: f"{x['name']}: {x['description']}",
    )
    
    command_id = command_info["id"]
    st.sidebar.info(f"Selected: **{command_info['name']}**\n\n{command_info['description']}")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["Configuration Builder", "YAML Output", "Command Execution"])
    
    with tab1:
        st.header("Configuration Builder")
        st.info("Fill in the form below to build your configuration.")
        
        # Build configuration from schema
        config = build_config_from_schema(schema)
        
        # Store config in session state
        st.session_state["config"] = config
    
    with tab2:
        st.header("YAML Output")
        
        if "config" in st.session_state:
            config = st.session_state["config"]
            yaml_output = yaml.dump(config, default_flow_style=False, sort_keys=False)
            
            st.subheader("Generated YAML Configuration")
            st.code(yaml_output, language="yaml")
            
            # Download button
            st.download_button(
                label="Download YAML File",
                data=yaml_output,
                file_name="config.yaml",
                mime="text/yaml",
            )
            
            # Save path input
            output_path = st.text_input(
                "Save Configuration To",
                value="./config_generated.yaml",
                key="output_path",
            )
            
            if st.button("Save to File"):
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(yaml_output)
                    st.success(f"Configuration saved to {output_path}")
                except Exception as e:
                    st.error(f"Error saving file: {e}")
        else:
            st.warning("Please go to the 'Configuration Builder' tab to create a configuration first.")
    
    with tab3:
        st.header("Command Execution")
        
        if "config" in st.session_state:
            config = st.session_state["config"]
            output_path = st.text_input(
                "Configuration File Path",
                value="./config_generated.yaml",
                key="exec_output_path",
            )

            st.subheader("Command Arguments")
            extra_args = render_command_args(command_id)

            # Build CLI command
            cli_command = build_cli_command(command_id, config, output_path, extra_args=extra_args)
            
            st.subheader("Generated CLI Command")
            st.code(cli_command, language="bash")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy to Clipboard"):
                    st.code(cli_command, language="bash")
                    st.success("Command displayed above - copy it manually")
            
            with col2:
                st.write("### Run with nohup")
                log_file = st.text_input(
                    "Log File Path",
                    value="./nohup_web.out",
                    key="log_file",
                )
                
                if st.button("▶️ Run with nohup"):
                    try:
                        process = run_command_with_nohup(cli_command, log_file)
                        st.success(f"Command started with nohup. Check logs at: {log_file}")
                        st.info(f"Process ID: {process.pid}")
                    except Exception as e:
                        st.error(f"Error running command: {e}")
            
            # Command options
            st.subheader("Command Options")
            show_full_command = st.checkbox("Show Full Command", value=False, key="show_full")
            
            if show_full_command:
                full_command = f"nohup {cli_command} > {st.session_state.get('log_file', './nohup_web.out')} 2>&1 &"
                st.code(full_command, language="bash")
        else:
            st.warning("Please go to the 'Configuration Builder' tab to create a configuration first.")


if __name__ == "__main__":
    main()
