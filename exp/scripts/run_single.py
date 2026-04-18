"""Train and evaluate a single model using RecBole.

Usage:
    python scripts/run_single.py --model BERT4Rec --config configs/bert4rec.yaml --seed 42
    python scripts/run_single.py --model DurationAwareBERT4Rec --config configs/variant_a.yaml
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger, get_model

# Custom model registry
CUSTOM_MODELS = {}


def _load_custom_models():
    """Lazy-load custom model classes."""
    global CUSTOM_MODELS
    if CUSTOM_MODELS:
        return
    try:
        from src.models.variant_a import VariantA
        CUSTOM_MODELS["DurationAwareBERT4Rec"] = VariantA
    except ImportError as e:
        print(f"Warning: Could not import Variant A: {e}")
    try:
        from src.models.variant_b import VariantB
        CUSTOM_MODELS["CounterfactualBERT4Rec"] = VariantB
    except ImportError as e:
        print(f"Warning: Could not import Variant B: {e}")
    try:
        from src.models.variant_c import VariantC
        CUSTOM_MODELS["CalibratedBERT4Rec"] = VariantC
    except ImportError as e:
        print(f"Warning: Could not import Variant C: {e}")


def run(model_name: str, config_files: list, seed: int = 42, dataset_name: str = "vod_data"):
    """Train and evaluate a single model."""
    _load_custom_models()
    is_custom = model_name in CUSTOM_MODELS

    # For custom models, use BERT4Rec as the base config model
    base_model = "BERT4Rec" if is_custom else model_name
    config_dict = {"seed": seed, "reproducibility": True}

    # Pop model uses float64 which MPS doesn't support — force CPU
    if model_name == "Pop":
        config_dict["gpu_id"] = ""

    # Temporarily clear sys.argv so RecBole doesn't warn about unknown CLI args
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    config = Config(
        model=base_model,
        dataset=dataset_name,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    sys.argv = saved_argv

    if is_custom:
        config["model"] = model_name

    init_seed(config["seed"], config["reproducibility"])

    # Reset root logger handlers so each model gets its own clean log file
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    init_logger(config)

    # Create dataset and dataloaders
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Create model
    if is_custom:
        model = CUSTOM_MODELS[model_name](config, dataset).to(config["device"])
    else:
        model = get_model(base_model)(config, dataset).to(config["device"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Model: {model_name} | Params: {n_params:,} | Seed: {seed}")
    print(f"Device: {config['device']} | Epochs: {config['epochs']}")
    print(f"{'='*60}\n")

    # Train
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # Evaluate on test
    test_result = trainer.evaluate(test_data)

    # Collect results
    result = {
        "model": model_name,
        "seed": seed,
        "n_params": n_params,
        "best_valid_score": float(best_valid_score),
        "valid_result": {k: float(v) for k, v in best_valid_result.items()},
        "test_result": {k: float(v) for k, v in test_result.items()},
    }

    # Save results
    out_dir = os.path.join("recbole_outputs", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train a single RecBole model")
    parser.add_argument("--model", required=True,
                        help="Model name (Pop, SASRec, BERT4Rec, DurationAwareBERT4Rec, etc.)")
    parser.add_argument("--config", nargs="+", default=["configs/base.yaml"],
                        help="Config file(s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="vod_data")
    args = parser.parse_args()

    result = run(args.model, args.config, args.seed, args.dataset)
    print(f"\nTest NDCG@10: {result['test_result'].get('ndcg@10', 'N/A'):.6f}")


if __name__ == "__main__":
    main()
