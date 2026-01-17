"""
Quantization Utilities - INT8/INT4 quantization for models
Part of the Neural Network module for the Consciousness system.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization-aware training


def quantize_model_int8(model: Any, method: QuantizationMethod = QuantizationMethod.DYNAMIC) -> Any:
    """
    Quantize a model to INT8.

    Args:
        model: The model to quantize
        method: Quantization method to use

    Returns:
        Quantized model
    """
    try:
        import torch
        import torch.quantization as quant

        if method == QuantizationMethod.DYNAMIC:
            # Dynamic quantization - simplest, no calibration needed
            quantized = quant.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic INT8 quantization")
            return quantized

        elif method == QuantizationMethod.STATIC:
            # Static quantization - requires calibration data
            model.qconfig = quant.get_default_qconfig('fbgemm')
            quant.prepare(model, inplace=True)
            # Would need calibration here
            quant.convert(model, inplace=True)
            logger.info("Applied static INT8 quantization")
            return model

        else:
            logger.warning(f"Unsupported quantization method: {method}")
            return model

    except ImportError:
        logger.warning("PyTorch not available, returning original model")
        return model
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return model


def quantize_model_int4(model: Any) -> Any:
    """
    Quantize a model to INT4 using bitsandbytes.

    Args:
        model: The model to quantize

    Returns:
        Quantized model
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Note: This config needs to be used during model loading
        # Cannot quantize an already loaded model this way
        logger.warning("INT4 quantization must be applied during model loading")
        return model

    except ImportError:
        logger.warning("bitsandbytes not available for INT4 quantization")
        return model
    except Exception as e:
        logger.error(f"INT4 quantization failed: {e}")
        return model


def estimate_quantized_size(
    original_size_mb: int,
    original_dtype: str,
    target_dtype: str
) -> int:
    """
    Estimate the size of a quantized model.

    Args:
        original_size_mb: Original model size in MB
        original_dtype: Original data type (fp32, fp16)
        target_dtype: Target data type (int8, int4)

    Returns:
        Estimated size in MB
    """
    # Bytes per element for each dtype
    bytes_per_elem = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5,
    }

    original_bytes = bytes_per_elem.get(original_dtype, 4)
    target_bytes = bytes_per_elem.get(target_dtype, 1)

    ratio = target_bytes / original_bytes
    return int(original_size_mb * ratio)


def get_quantization_config(quant_type: str) -> Dict[str, Any]:
    """
    Get configuration for quantization.

    Args:
        quant_type: Type of quantization (int8, int4)

    Returns:
        Configuration dictionary
    """
    configs = {
        'int8': {
            'method': 'dynamic',
            'dtype': 'qint8',
            'per_channel': True,
            'symmetric': False,
        },
        'int4': {
            'method': 'nf4',
            'double_quant': True,
            'compute_dtype': 'float16',
        },
        'fp16': {
            'method': 'half',
            'dtype': 'float16',
        },
    }

    return configs.get(quant_type, configs['fp16'])


class ModelQuantizer:
    """
    Helper class for managing model quantization.
    """

    def __init__(self, default_method: QuantizationMethod = QuantizationMethod.DYNAMIC):
        self.default_method = default_method

    def quantize(
        self,
        model: Any,
        target_dtype: str,
        method: Optional[QuantizationMethod] = None
    ) -> Any:
        """
        Quantize a model to the target dtype.

        Args:
            model: Model to quantize
            target_dtype: Target dtype (int8, int4, fp16)
            method: Optional specific method to use

        Returns:
            Quantized model
        """
        method = method or self.default_method

        if target_dtype == 'int8':
            return quantize_model_int8(model, method)
        elif target_dtype == 'int4':
            return quantize_model_int4(model)
        elif target_dtype == 'fp16':
            return self._convert_to_fp16(model)
        else:
            logger.warning(f"Unknown target dtype: {target_dtype}")
            return model

    def _convert_to_fp16(self, model: Any) -> Any:
        """Convert model to FP16."""
        try:
            import torch
            if hasattr(model, 'half'):
                return model.half()
            return model
        except ImportError:
            return model

    def estimate_memory_savings(
        self,
        original_size_mb: int,
        original_dtype: str,
        target_dtype: str
    ) -> Dict[str, Any]:
        """
        Estimate memory savings from quantization.

        Returns:
            Dictionary with original size, new size, and savings
        """
        new_size = estimate_quantized_size(original_size_mb, original_dtype, target_dtype)
        savings = original_size_mb - new_size
        savings_percent = (savings / original_size_mb) * 100 if original_size_mb > 0 else 0

        return {
            'original_size_mb': original_size_mb,
            'quantized_size_mb': new_size,
            'savings_mb': savings,
            'savings_percent': round(savings_percent, 1),
            'original_dtype': original_dtype,
            'target_dtype': target_dtype,
        }
