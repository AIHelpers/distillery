"""Distillery local fine-tuning worker.

Go launches this package as a subprocess for each training job:

    python -m trainer.run --job-dir ... --config ... --dataset ...
"""

__version__ = "0.1.0"