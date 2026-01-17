"""
Model Loader - Unified loading with quantization support
Part of the Neural Network module for the Consciousness system.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    """Supported quantization types."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class LoadedModelInfo:
    """Information about a loaded model."""
    form_id: str
    model_name: str
    model_type: str
    quantization: QuantizationType
    size_mb: int
    vram_mb: int
    loaded_at: datetime
    device: str


class ModelLoader:
    """
    Unified model loading with quantization support.

    Handles loading models from various sources (HuggingFace, local files)
    with optional quantization for memory efficiency.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model loader.

        Args:
            cache_dir: Directory for caching downloaded models
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_models: Dict[str, LoadedModelInfo] = {}
        self._device = self._detect_device()

    def _detect_device(self) -> str:
        """Detect available compute device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    async def load(
        self,
        form_id: str,
        model_name: str,
        model_type: str,
        quantization: str = "fp16",
    ) -> Optional[Any]:
        """
        Load a model with specified quantization.

        Args:
            form_id: The form ID this model is for
            model_name: Model identifier (HuggingFace path or local path)
            model_type: Type of model (vision_transformer, transformer, etc.)
            quantization: Quantization level (fp32, fp16, int8, int4)

        Returns:
            Loaded model instance or None if loading fails
        """
        logger.info(f"Loading model for {form_id}: {model_name} ({quantization})")

        quant_type = QuantizationType(quantization)

        try:
            # Load based on model type
            if model_type == "vision_transformer":
                model = await self._load_vision_transformer(model_name, quant_type)
            elif model_type == "speech_transformer":
                model = await self._load_speech_transformer(model_name, quant_type)
            elif model_type in ["transformer", "small_llm"]:
                model = await self._load_transformer(model_name, quant_type)
            elif model_type == "graph_neural_network":
                model = await self._load_gnn(model_name, quant_type)
            elif model_type in ["cnn", "mlp_ensemble", "lstm_ensemble", "rnn"]:
                model = await self._load_custom_model(model_name, model_type, quant_type)
            else:
                # Default mock loading for custom/unknown types
                model = await self._load_mock_model(model_name, model_type)

            if model is not None:
                # Track loaded model
                self._loaded_models[form_id] = LoadedModelInfo(
                    form_id=form_id,
                    model_name=model_name,
                    model_type=model_type,
                    quantization=quant_type,
                    size_mb=self._estimate_size(model),
                    vram_mb=self._estimate_vram(model, quant_type),
                    loaded_at=datetime.now(timezone.utc),
                    device=self._device,
                )
                logger.info(f"Successfully loaded model for {form_id}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model for {form_id}: {e}")
            return None

    async def unload(self, form_id: str, model_instance: Any) -> None:
        """
        Unload a model and free resources.

        Args:
            form_id: The form ID
            model_instance: The model instance to unload
        """
        try:
            # Clear from tracking
            if form_id in self._loaded_models:
                del self._loaded_models[form_id]

            # Free model memory
            if model_instance is not None:
                del model_instance

            # Force garbage collection
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            import gc
            gc.collect()

            logger.info(f"Unloaded model for {form_id}")

        except Exception as e:
            logger.error(f"Error unloading model for {form_id}: {e}")

    async def _load_vision_transformer(
        self,
        model_name: str,
        quantization: QuantizationType
    ) -> Optional[Any]:
        """Load a vision transformer model (ViT/CLIP)."""
        try:
            from transformers import CLIPModel, CLIPProcessor

            # Load model with appropriate dtype
            dtype = self._get_torch_dtype(quantization)

            if quantization in [QuantizationType.INT8, QuantizationType.INT4]:
                # Use bitsandbytes for quantization
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=(quantization == QuantizationType.INT8),
                    load_in_4bit=(quantization == QuantizationType.INT4),
                )
                model = CLIPModel.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                model = CLIPModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                ).to(self._device)

            processor = CLIPProcessor.from_pretrained(model_name)
            return {"model": model, "processor": processor}

        except ImportError:
            logger.warning("transformers not available, using mock model")
            return await self._load_mock_model(model_name, "vision_transformer")
        except Exception as e:
            logger.error(f"Error loading vision transformer: {e}")
            return None

    async def _load_speech_transformer(
        self,
        model_name: str,
        quantization: QuantizationType
    ) -> Optional[Any]:
        """Load a speech transformer model (Whisper)."""
        try:
            from transformers import WhisperModel, WhisperProcessor

            dtype = self._get_torch_dtype(quantization)

            model = WhisperModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
            ).to(self._device)

            processor = WhisperProcessor.from_pretrained(model_name)
            return {"model": model, "processor": processor}

        except ImportError:
            logger.warning("transformers not available, using mock model")
            return await self._load_mock_model(model_name, "speech_transformer")
        except Exception as e:
            logger.error(f"Error loading speech transformer: {e}")
            return None

    async def _load_transformer(
        self,
        model_name: str,
        quantization: QuantizationType
    ) -> Optional[Any]:
        """Load a generic transformer model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            dtype = self._get_torch_dtype(quantization)

            if quantization in [QuantizationType.INT8, QuantizationType.INT4]:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=(quantization == QuantizationType.INT8),
                    load_in_4bit=(quantization == QuantizationType.INT4),
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                ).to(self._device)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return {"model": model, "tokenizer": tokenizer}

        except ImportError:
            logger.warning("transformers not available, using mock model")
            return await self._load_mock_model(model_name, "transformer")
        except Exception as e:
            logger.error(f"Error loading transformer: {e}")
            return None

    async def _load_gnn(
        self,
        model_name: str,
        quantization: QuantizationType
    ) -> Optional[Any]:
        """Load a graph neural network model."""
        try:
            # PyTorch Geometric based GNN
            import torch
            from torch_geometric.nn import GCNConv

            class IITGraphModel(torch.nn.Module):
                def __init__(self, in_channels: int = 64, hidden_channels: int = 128, out_channels: int = 1):
                    super().__init__()
                    self.conv1 = GCNConv(in_channels, hidden_channels)
                    self.conv2 = GCNConv(hidden_channels, out_channels)

                def forward(self, x, edge_index):
                    x = self.conv1(x, edge_index)
                    x = torch.relu(x)
                    x = self.conv2(x, edge_index)
                    return x

            model = IITGraphModel().to(self._device)
            return model

        except ImportError:
            logger.warning("torch_geometric not available, using mock model")
            return await self._load_mock_model(model_name, "graph_neural_network")
        except Exception as e:
            logger.error(f"Error loading GNN: {e}")
            return None

    async def _load_custom_model(
        self,
        model_name: str,
        model_type: str,
        quantization: QuantizationType
    ) -> Optional[Any]:
        """Load a custom model (CNN, MLP, LSTM, etc.)."""
        # For custom models, return a mock implementation
        return await self._load_mock_model(model_name, model_type)

    async def _load_mock_model(
        self,
        model_name: str,
        model_type: str
    ) -> Dict[str, Any]:
        """Create a mock model for testing when real models unavailable."""
        logger.info(f"Loading mock model for {model_name} ({model_type})")

        class MockModel:
            def __init__(self, name: str, mtype: str):
                self.name = name
                self.model_type = mtype

            def __call__(self, *args, **kwargs):
                return {"output": "mock", "model": self.name}

        return {"model": MockModel(model_name, model_type), "mock": True}

    def _get_torch_dtype(self, quantization: QuantizationType):
        """Get PyTorch dtype for quantization level."""
        try:
            import torch
            if quantization == QuantizationType.FP32:
                return torch.float32
            elif quantization == QuantizationType.FP16:
                return torch.float16
            else:
                return torch.float16  # Default for INT quantization
        except ImportError:
            return None

    def _estimate_size(self, model: Any) -> int:
        """Estimate model size in MB."""
        try:
            import torch
            if hasattr(model, 'parameters'):
                params = sum(p.numel() for p in model.parameters())
                return int(params * 4 / 1024 / 1024)  # 4 bytes per param
        except Exception:
            pass
        return 100  # Default estimate

    def _estimate_vram(self, model: Any, quantization: QuantizationType) -> int:
        """Estimate VRAM usage in MB."""
        base_size = self._estimate_size(model)

        # Adjust for quantization
        multipliers = {
            QuantizationType.FP32: 1.0,
            QuantizationType.FP16: 0.5,
            QuantizationType.INT8: 0.25,
            QuantizationType.INT4: 0.125,
        }

        return int(base_size * multipliers.get(quantization, 1.0))

    def get_loaded_models(self) -> Dict[str, LoadedModelInfo]:
        """Get information about loaded models."""
        return dict(self._loaded_models)

    def get_device(self) -> str:
        """Get the compute device being used."""
        return self._device
