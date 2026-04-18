"""
Comprehensive experiment logging module with TensorBoard, console, and JSON support.

Provides:
- ExperimentLogger: Unified logging to TensorBoard, console, and JSON
- setup_logging: Configure console and file logging
- Tracks metrics, hyperparameters, and experiment metadata
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union

from torch.utils.tensorboard import SummaryWriter


def setup_logging(
    output_dir: Union[str, Path],
    model_name: str = "experiment",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up console and file logging for an experiment.

    Creates both a console logger (stdout) and a file logger. The console logger
    uses INFO level and includes timestamps. The file logger captures all messages.

    Args:
        output_dir: Directory where to save log files. Created if it doesn't exist.
        model_name: Name of the model/experiment for log file naming.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logger instance for use in experiments.

    Examples:
        >>> logger = setup_logging('outputs/exp1', model_name='resnet50')
        >>> logger.info("Training started")

    Notes:
        - Logger name is 'experiment' (shared across calls)
        - Previous handlers are cleared to avoid duplicate logs
        - Log files are saved as {model_name}.log
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get or create logger
    logger = logging.getLogger("experiment")
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    # Define log format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    log_file = output_dir / f"{model_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")
    return logger


class ExperimentLogger:
    """
    Unified experiment logger combining TensorBoard, JSON, and console logging.

    Tracks:
    - Scalars and metrics via TensorBoard
    - Hyperparameters and final results
    - Epoch-by-epoch results in JSON format for analysis
    - Text logs and arbitrary data

    Attributes:
        tb_writer: TensorBoard SummaryWriter instance
        output_dir: Directory for saving outputs
        model_name: Name of the model/experiment
        run_log_path: Path to JSON log file
        run_log: In-memory experiment log dictionary
        logger: Python logger instance

    Examples:
        >>> exp_logger = ExperimentLogger(
        ...     tb_dir='outputs/runs/exp1',
        ...     output_dir='outputs/exp1',
        ...     model_name='resnet50'
        ... )
        >>> exp_logger.log_scalar('loss/train', 0.5, step=0)
        >>> exp_logger.log_epoch(0, {'loss': 0.5}, {'loss': 0.4})
        >>> exp_logger.log_final({'accuracy': 0.95})
        >>> exp_logger.close()
    """

    def __init__(
        self,
        tb_dir: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: str = "experiment",
    ):
        """
        Initialize ExperimentLogger.

        Args:
            tb_dir: Directory for TensorBoard event files.
            output_dir: Directory for all output files (logs, configs, etc.).
            model_name: Name of the model for file naming and logging.
        """
        self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.run_log_path = self.output_dir / "run_log.json"

        # Initialize run log with metadata
        self.run_log: Dict[str, Any] = {
            "model_name": model_name,
            "started_at": datetime.now().isoformat(),
            "epochs": [],
        }

        self.logger = logging.getLogger("experiment")
        self.logger.info(f"ExperimentLogger initialized for {model_name}")
        self.logger.info(f"TensorBoard: {tb_dir}")
        self.logger.info(f"Outputs: {output_dir}")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard.

        Args:
            tag: Name of the scalar (e.g., 'loss/train', 'metrics/accuracy').
            value: Scalar value to log.
            step: Training step/iteration number.

        Examples:
            >>> logger.log_scalar('loss/train', 0.45, step=100)
            >>> logger.log_scalar('metrics/accuracy', 0.92, step=100)
        """
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        tag_value_dict: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Log multiple scalar values to TensorBoard at once.

        Useful for logging multiple metrics from the same step together.

        Args:
            tag_value_dict: Dictionary mapping metric names to values.
            step: Training step/iteration number.
            prefix: Optional prefix for all tags (e.g., 'train' for 'train/loss').

        Examples:
            >>> logger.log_scalars({
            ...     'loss': 0.45,
            ...     'accuracy': 0.92,
            ... }, step=100, prefix='train')
        """
        for tag, value in tag_value_dict.items():
            full_tag = f"{prefix}/{tag}" if prefix else tag
            self.tb_writer.add_scalar(full_tag, value, step)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log hyperparameters and final metrics to TensorBoard.

        Creates a hparam experiment page in TensorBoard for easy comparison of runs.

        Args:
            hparams: Dictionary of hyperparameters (e.g., learning rate, batch size).
            metrics: Dictionary of final metrics to compare across hparam sets.

        Examples:
            >>> logger.log_hparams(
            ...     {'lr': 1e-3, 'batch_size': 32},
            ...     {'final_accuracy': 0.95}
            ... )

        Notes:
            - Only call once per experiment (typically at the end)
            - Metrics should be scalar values
        """
        try:
            self.tb_writer.add_hparams(hparams, metrics)
        except Exception as e:
            self.logger.warning(f"Failed to log hparams: {e}")

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        """
        Log arbitrary text to TensorBoard.

        Useful for logging model architecture, configs, or analysis results.

        Args:
            tag: Name for the text (e.g., 'model/architecture').
            text: Text content to log.
            step: Optional step number (default 0).

        Examples:
            >>> logger.log_text('config', config_yaml_string, step=0)
            >>> logger.log_text('model/summary', model_summary, step=0)
        """
        self.tb_writer.add_text(tag, text, step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        valid_metrics: Dict[str, float],
    ) -> None:
        """
        Log epoch results to JSON file.

        Accumulates epoch results for post-training analysis. Automatically saves
        to run_log.json after each epoch.

        Args:
            epoch: Epoch number (0-indexed).
            train_metrics: Dictionary of training metrics for this epoch.
            valid_metrics: Dictionary of validation metrics for this epoch.

        Examples:
            >>> logger.log_epoch(0, {'loss': 0.45}, {'loss': 0.42, 'accuracy': 0.91})
            >>> logger.log_epoch(1, {'loss': 0.35}, {'loss': 0.38, 'accuracy': 0.93})

        Notes:
            - Automatically saved to run_log.json
            - Timestamps are included for each epoch
        """
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": train_metrics,
            "valid": valid_metrics,
        }
        self.run_log["epochs"].append(entry)
        self._save_json()
        self.logger.debug(f"Logged epoch {epoch}")

    def log_final(
        self,
        test_metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log final test results and optionally save config.

        Called at the end of training to record final results and metadata.
        Automatically saves to run_log.json.

        Args:
            test_metrics: Dictionary of final test metrics.
            config: Optional config dictionary to save with results.

        Examples:
            >>> logger.log_final(
            ...     {'test_accuracy': 0.94, 'test_loss': 0.18},
            ...     config={'model': 'resnet50', 'lr': 1e-3}
            ... )

        Notes:
            - Sets 'finished_at' timestamp
            - Includes config for reproducibility
        """
        self.run_log["finished_at"] = datetime.now().isoformat()
        self.run_log["test_metrics"] = test_metrics
        if config:
            self.run_log["config"] = config
        self._save_json()
        self.logger.info("Final metrics logged")

    def _save_json(self) -> None:
        """
        Save the in-memory run log to JSON file.

        Internal method called automatically after logging epochs or final results.
        Uses 'default=str' to handle non-serializable types (e.g., numpy types).
        """
        try:
            with open(self.run_log_path, 'w') as f:
                json.dump(self.run_log, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save run log to {self.run_log_path}: {e}")

    def get_run_log(self) -> Dict[str, Any]:
        """
        Get the current in-memory run log.

        Useful for debugging or checking logged data without reading JSON.

        Returns:
            Dictionary containing all logged data.

        Examples:
            >>> log = logger.get_run_log()
            >>> print(log['epochs'][-1])
        """
        return self.run_log

    def close(self) -> None:
        """
        Close the TensorBoard writer and finalize logging.

        Should be called at the end of training to ensure all data is flushed
        to disk. Safe to call multiple times.

        Examples:
            >>> try:
            ...     # ... training code ...
            ... finally:
            ...     logger.close()
        """
        self.tb_writer.close()
        self.logger.info("ExperimentLogger closed")
