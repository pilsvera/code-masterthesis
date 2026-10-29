import glob
import json
import logging
import yaml
import os

logger = logging.getLogger(__name__)

RULES = [
    {
        "if": {"source_type": "source_1"},
        "then": {"mean_std": "CELLPAINTING_1"},
        "message": "If source_type is 'source_1', mean_std must be 'CELLPAINTING_1'."
    },
    {
        "if": {"source_type": "source_3"},
        "then": {"mean_std": "CELLPAINTING_3"},
        "message": "If source_type is 'source_3', mean_std must be 'CELLPAINTING_3'."
    },
    {
        "if": {"source_type": "source_1_w_neg"},
        "then": {"mean_std": "CELLPAINTING_1_w_NEG"},
        "message": "If source_type is 'source_1_w_neg', mean_std must be 'CELLPAINTING_1_w_NEG'."
    },
    {
        "if": {"source_type": "source_3_w_neg"},
        "then": {"mean_std": "CELLPAINTING_3_w_NEG"},
        "message": "If source_type is 'source_3_w_neg', mean_std must be 'CELLPAINTING_3_w_NEG'."
    },
    {
        "if_nested": {"architecture": "ResNet50_Modified"},
        "then_nested": {"augmentation.resize": [224, 224]},
        "message": "If architecture is 'ResNet50_Modified', resize must be [224, 224]."
    },
    {
        "if_nested": {"architecture": "OpenPhenomMAE"},
        "then_nested": {"augmentation.resize": [256, 256]},
        "message": "If architecture is 'OpenPhenomMAE', resize must be [256, 256]."
    },
    {
        "if": {"resume": False},
        "then": {"checkpoint": ""},
        "message": "If resume is False, checkpoint must be empty.",
        "silent": True
    },
    {
        "if": {"embedding_mode": False},
        "then": {"embeddings_path": ""},
        "message": "If embedding_mode is False, embeddings_path must be empty.",
        "silent": True
    },
    # lr_warmup_epochs only applies to the cosine scheduler.
    # ScheduleFree ('auto') manages its own warmup internally; the config param has no effect.
    # 'step' scheduler has no warmup implementation either.
    {
        "if": {"lr_scheduler": "auto"},
        "then": {"lr_warmup_epochs": 0},
        "message": "lr_warmup_epochs has no effect with lr_scheduler='auto' (ScheduleFree). Resetting to 0.",
        "silent": True
    },
    {
        "if": {"lr_scheduler": "step"},
        "then": {"lr_warmup_epochs": 0},
        "message": "lr_warmup_epochs is not implemented for lr_scheduler='step'. Resetting to 0.",
        "silent": True
    },
    # In embedding mode the model never sees raw images, so returning channelwise
    # embeddings from the image encoder is meaningless.
    {
        "if": {"embedding_mode": True},
        "then": {"return_channelwise_embeddings": False},
        "message": "return_channelwise_embeddings has no effect in embedding_mode. Setting to False.",
        "silent": True
    }
]


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_saved_config(path):
    import re
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.constructor.ConstructorError:
        # Old wandb config dump with Python object tags — strip tags and retry
        with open(path, "r") as f:
            raw = f.read()
        raw = re.sub(r"![!]?\S+", "", raw)
        data = yaml.safe_load(raw)
    cfg = data.get("config", data)
    # Old wandb dumps wrap the real hyperparameters inside a Config object whose
    # values live one level deeper, under `_items`; the new clean format stores
    # them directly under `config`. Normalize both to a flat dict so consumers can
    # read config["splits"], config["augmentation"], etc. regardless of which run
    # produced the YAML.
    if isinstance(cfg, dict) and "splits" not in cfg and isinstance(cfg.get("_items"), dict):
        cfg = cfg["_items"]
    return cfg


def get_nested(config, key_path):
    """Access nested dictionary with dot notation."""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def set_nested(config, key_path, value):
    """Set nested dictionary value with dot notation."""
    keys = key_path.split('.')
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def check_resume(config):
    """Check resume-related config. Returns list of error strings (not auto-fixable)."""
    errors = []
    if not config.get("resume", False):
        return []
    if not config.get("checkpoint", False):
        errors.append("Resume is True but checkpoint_path is not set.")
        return errors

    checkpoint_path = config["checkpoint"]
    # Support both flat (dir/run_id/model.pth) and grouped
    # (dir/group/run_id/model.pth) checkpoint layouts.
    run_dir = os.path.dirname(checkpoint_path)
    if not run_dir:
        errors.append(f"Checkpoint path has unexpected format: {checkpoint_path}")
        return errors

    keys_path = os.path.join(run_dir, "split_keys.json")
    if not os.path.exists(keys_path):
        errors.append(f"split_keys.json not found at {keys_path}")
        return errors

    with open(keys_path) as f:
        keys = json.load(f)
    if keys["split_strategy"] != config["splits"]:
        errors.append(f"Split strategy does not match. "
                      f"It should be {keys['split_strategy']} not {config['splits']}")

    saved_config_files = glob.glob(os.path.join(run_dir, "*_config.yaml"))
    if len(saved_config_files) > 1:
        errors.append(f"Multiple config files found in {run_dir}. "
                      "Please specify the correct one.")
    elif len(saved_config_files) == 0:
        errors.append(f"No config file found in {run_dir}.")
    else:
        saved_config = load_saved_config(saved_config_files[0])
        # Keys that are expected to differ between training and inference
        skip_keys = {"checkpoint", "inference", "resume", "num_workers", "augmentation"}
        # In inference / embedding-dump mode, training-only hyperparams can
        # legitimately differ from the training run — only architecture, splits,
        # seed, data source, normalisation, and resize need to match.
        if config.get("inference") or config.get("return_channelwise_embeddings"):
            skip_keys |= {
                "return_channelwise_embeddings",
                "batch_size", "epochs",
                "lr", "lr_scheduler", "lr_warmup_epochs",
                "weight_decay", "label_smoothing",
                "early_stopping_metric", "early_stopping_patience",
                # freeze only controls encoder requires_grad during training;
                # the forward pass (and thus extracted embeddings) is identical
                # frozen or not, so it may differ from the training run here.
                "freeze",
            }
            # eval_split / test_split are NOT skipped — they affect which
            # samples land in train/val/test and must match the training run.
        common_keys = (set(config.keys()) & set(saved_config.keys())) - skip_keys
        for key in sorted(common_keys):
            if config[key] != saved_config[key]:
                errors.append(f"Config mismatch for {key}. "
                              f"Expected {saved_config[key]}, got {config[key]}.")
    return errors


def check_and_fix_rules(config):
    """Check rules and auto-fix violations. Modifies config in-place.

    Returns:
        changes: list of strings describing what was auto-fixed
        errors: list of strings for issues that cannot be auto-fixed
    """
    changes = []
    errors = []

    for rule in RULES:
        if_condition = rule.get("if", {})
        if_nested_condition = rule.get("if_nested", {})
        then_condition = rule.get("then", {})
        then_nested_condition = rule.get("then_nested", {})

        # Check if the "if" condition matches
        if_matches = (
            all(config.get(key) == value for key, value in if_condition.items()) and
            all(get_nested(config, key) == value for key, value in if_nested_condition.items())
        )
        if not if_matches:
            continue

        silent = rule.get("silent", False)

        # Auto-fix flat "then" conditions
        for key, expected in then_condition.items():
            actual = config.get(key)
            if actual != expected:
                config[key] = expected
                if not silent:
                    msg = (f"AUTO-FIX: {rule['message']} "
                           f"Changed {key}: {actual!r} -> {expected!r}")
                    changes.append(msg)

        # Auto-fix nested "then" conditions
        for key_path, expected in then_nested_condition.items():
            actual = get_nested(config, key_path)
            if actual != expected:
                set_nested(config, key_path, expected)
                if not silent:
                    msg = (f"AUTO-FIX: {rule['message']} "
                           f"Changed {key_path}: {actual!r} -> {expected!r}")
                    changes.append(msg)

    # Resume checks are not auto-fixable
    resume_errors = check_resume(config)
    errors.extend(resume_errors)

    return changes, errors


def validate_config(config):
    """Run all checks on config. Auto-fixes what it can, raises on hard errors.

    Modifies config in-place. Logs all changes and errors.

    Returns:
        config: the (possibly modified) config dict
    """
    changes, errors = check_and_fix_rules(config)

    for msg in changes:
        logger.warning(msg)
        print(f"[CONFIG CHECK] {msg}")

    if errors:
        for msg in errors:
            logger.error(f"[CONFIG ERROR] {msg}")
            print(f"[CONFIG ERROR] {msg}")
        raise ValueError(
            "Config validation failed with errors that cannot be auto-fixed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    if not changes:
        print("[CONFIG CHECK] All checks passed.")

    return config


def main(config_path):
    config = load_config(config_path)
    validate_config(config)


if __name__ == "__main__":
    config_path = "wandb_config.yaml"
    main(config_path)
