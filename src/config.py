"""Minimal config system with value propagation via OmegaConf interpolation."""

from omegaconf import OmegaConf, DictConfig

from src.model.recurrent_lstm_cell import mLSTMWeavingCellConfig
from src.model.recurrent_lstm_layer import mLSTMWeavingLayerConfig
from src.tsf import WeavingBlockLSTMConfig, ModelConfig, TrainingConfig


def make_config(
    num_heads: int = 4,
    embedding_dim: int = 32,
    head_embedding_dim: int = 8,
    n_layers: int = 4,
    **kw
) -> DictConfig:
    """
    Create a config where top-level values auto-propagate to all nested configs.
    
    Just change num_heads here and it flows everywhere.
    """
    quantiles = kw.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    cfg = OmegaConf.create({
        # Top-level values (single source of truth)
        "num_heads": num_heads,
        "embedding_dim": embedding_dim,
        "head_embedding_dim": head_embedding_dim,
        "n_layers": n_layers,
        "input_dim": kw.get("input_dim", 1),
        "output_dim": len(quantiles),
        "dropout": kw.get("dropout", 0.1),
        "dtype": kw.get("dtype", "bfloat16"),
        
        # Training
        "training": {
            "quantiles": quantiles,
            "learning_rate": kw.get("learning_rate", 1e-3),
            "weight_decay": kw.get("weight_decay", 0.0),
            "batch_size": kw.get("batch_size", 32),
        },
        
        # Nested configs - use ${...} to reference top-level values
        "weaving_block": {
            "n_layers": "${n_layers}",
            "embedding_dim": "${head_embedding_dim}",
            "num_heads": "${num_heads}",
            
            "layer": {
                "embedding_dim": "${head_embedding_dim}",
                "num_heads": "${num_heads}",
                "dropout": "${dropout}",
                "dtype": "${dtype}",
                "conv1d_kernel_size": kw.get("conv1d_kernel_size", 4),
                "qkv_proj_blocksize": kw.get("qkv_proj_blocksize", 4),
                "bias": kw.get("bias", False),
                
                "mlstm_cell": {
                    "embedding_dim": "${head_embedding_dim}",
                    "num_heads": "${num_heads}",
                    "dtype": "${dtype}",
                }
            }
        }
    })
    OmegaConf.resolve(cfg)  # Resolve all ${...} references
    return cfg


def load_config(path: str) -> DictConfig:
    """Load from YAML file."""
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    # Auto-set output_dim from quantiles if training section exists
    if "training" in cfg and "quantiles" in cfg.training:
        cfg.output_dim = len(cfg.training.quantiles)
    return cfg


def cfg_to_model_config(cfg):
    """Convert OmegaConf config to ModelConfig dataclass."""
    cell = mLSTMWeavingCellConfig(
        embedding_dim=cfg.head_embedding_dim * cfg.num_heads,  # inner dim
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
    )
    layer = mLSTMWeavingLayerConfig(
        embedding_dim=cfg.head_embedding_dim,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        dtype=cfg.dtype,
        mlstm_cell=cell,
    )
    block = WeavingBlockLSTMConfig(
        n_layers=cfg.n_layers,
        embedding_dim=cfg.head_embedding_dim,
        num_heads=cfg.num_heads,
        weaving_layer_config=layer,
    )
    return ModelConfig(
        input_dim=cfg.input_dim,
        embedding_dim=cfg.embedding_dim,
        head_embedding_dim=cfg.head_embedding_dim,
        num_heads=cfg.num_heads,
        n_layers=cfg.n_layers,
        output_dim=cfg.output_dim,
        weaving_block_config=block,
    )


def cfg_to_training_config(cfg):
    """Convert OmegaConf config to TrainingConfig dataclass."""
    model_config = cfg_to_model_config(cfg)

    training_cfg = cfg.training if "training" in cfg else cfg

    return TrainingConfig(
        learning_rate=training_cfg.get("learning_rate", 1e-3),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        dropout=cfg.dropout,
        quantiles=list(training_cfg.get("quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])),
        model_config=model_config,
        log_every=training_cfg.get("log_every", 10),
        num_epochs=training_cfg.get("num_epochs", 100),
        batch_size=training_cfg.get("batch_size", 32),
    )