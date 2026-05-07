import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

"""
Source Paper: Quantized Probabilistic AI for Gear Fault Diagnosis in Motor Drives
URL: http://arxiv.org/abs/2605.05032v1

Summary:
The paper proposes a Quantized Probabilistic Neural Network (QPNN) designed for 
on-device fault diagnosis. The core mathematical idea is to combine Bayesian 
uncertainty estimation (via Monte Carlo Dropout or Variational Inference) with 
Linear Quantization. 

By quantizing the weights and activations to low-precision integers (e.g., INT8), 
the engine reduces memory bandwidth and power consumption on edge hardware while 
maintaining a probabilistic output (mean and variance) to avoid overconfident 
misclassifications in industrial motor drive gear fault detection.

Key Hyperparameters:
- quantization_bits (int): Precision of the quantization (default: 8).
- dropout_rate (float): Probability of dropping units to simulate probabilistic 
  inference (default: 0.2).
- scale_factor (float): Scaling factor for mapping floats to integers.
"""

class QuantizedLinear(nn.Module):
    """
    A Linear layer that implements symmetric linear quantization for weights 
    and activations to simulate low-precision hardware execution.
    """
    def __init__(self, in_features: int, out_features: int, bits: int = 8):
        """
        Initialize the QuantizedLinear layer.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        bits : int
            The bit-width for quantization.
        """
        super(QuantizedLinear, self).__init__()
        self.bits = bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def quantize(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Perform symmetric linear quantization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to quantize.
        bits : int
            Bit-width.

        Returns
        -------
        torch.Tensor
            Quantized and dequantized tensor (fake quantization).
        """
        q_min = -(2**(bits - 1))
        q_max = (2**(bits - 1)) - 1
        
        # Calculate scale based on the absolute maximum of the tensor
        scale = x.abs().max() / q_max if x.abs().max() != 0 else 1.0
        
        # Quantize: Round(x / scale) and clamp to bit range
        x_q = torch.clamp(torch.round(x / scale), q_min, q_max)
        
        # Dequantize: Bring back to float for gradient flow (Fake Quantization)
        return x_q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight and activation quantization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear projection of quantized inputs and weights.
        """
        # Quantize weights and inputs to simulate the edge-device constraint
        q_weight = self.quantize(self.weight, self.bits)
        q_x = self.quantize(x, self.bits)
        
        return F.linear(q_x, q_weight, self.bias)

class ProbabilisticQuantizedNet(nn.Module):
    """
    Probabilistic Neural Network using Quantized layers and MC Dropout 
    for uncertainty-aware fault diagnosis.
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64, bits: int = 8, dropout_rate: float = 0.2):
        """
        Initialize the QPNN model.

        Parameters
        ----------
        input_dim : int
            Dimension of the input feature vector (e.g., vibration signals).
        num_classes : int
            Number of fault categories.
        hidden_dim : int
            Number of neurons in the hidden layer.
        bits : int
            Quantization bit-depth.
        dropout_rate : float
            Dropout probability for probabilistic inference.
        """
        super(ProbabilisticQuantizedNet, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Layer 1: Quantized Linear -> ReLU -> Dropout
        self.layer1 = QuantizedLinear(input_dim, hidden_dim, bits=bits)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Layer 2: Quantized Linear -> Softmax
        self.layer2 = QuantizedLinear(hidden_dim, num_classes, bits=bits)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass. Note: Dropout is active during inference for probabilistic estimation.

        Parameters
        ----------
        x : torch.Tensor
            Input signals.
        training : bool
            Whether the model is in training mode.

        Returns
        -------
        torch.Tensor
            Class probabilities (Logits).
        """
        x = F.relu(self.layer1(x))
        # Keep dropout active during evaluation to sample from the posterior (MC Dropout)
        x = self.dropout(x) 
        x = self.layer2(x)
        return x

def core_inference_engine(
    model: ProbabilisticQuantizedNet, 
    input_data: torch.Tensor, 
    num_samples: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The core inference engine implementing Quantized Probabilistic AI.
    It performs Monte Carlo (MC) sampling to generate a predictive 
    distribution, providing both the most likely fault class and 
    the associated uncertainty (variance).

    Parameters
    ----------
    model : ProbabilisticQuantizedNet
        The pre-trained quantized probabilistic model.
    input_data : torch.Tensor
        Input sensor data tensor of shape (batch_size, input_dim).
    num_samples : int
        Number of stochastic forward passes to estimate uncertainty.

        Returns
        -------
        predictions : torch.Tensor
            The mean predicted class probabilities for each sample.
        uncertainty : torch.Tensor
            The predictive variance (uncertainty) for each sample.
    """
    model.eval() # Set to eval, but dropout is handled manually in the model's forward logic
    
    # Storage for stochastic forward passes
    all_probs = []

    with torch.no_grad():
        for _ in range(num_samples):
            # We force the model to behave as if it is training for the dropout layers
            # to enable Monte Carlo sampling from the weight distribution
            model.train() 
            logits = model(input_data)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

    # Stack samples: (num_samples, batch_size, num_classes)
    stacked_probs = torch.stack(all_probs)
    
    # Calculate the predictive mean (Expected probability of each class)
    # Mathematical Idea: E[p] = (1/T) * sum(p_t)
    predictions = torch.mean(stacked_probs, dim=0)
    
    # Calculate the predictive variance (Uncertainty)
    # Mathematical Idea: Var(p) = E[p^2] - (E[p])^2
    # High variance indicates the model is uncertain about the gear fault
    variance = torch.var(stacked_probs, dim=0)
    uncertainty = torch.mean(variance, dim=-1) # Average uncertainty across classes per sample

    return predictions, uncertainty

if __name__ == "__main__":
    # Setup: Mock data for gear fault diagnosis
    # 10 samples, 20 vibration features, 3 fault types (Healthy, Outer Race, Inner Race)
    BATCH_SIZE = 10
    INPUT_DIM = 20
    NUM_CLASSES = 3
    
    torch.manual_seed(42)
    mock_input = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # Initialize the Quantized Probabilistic Network
    qpnn_model = ProbabilisticQuantizedNet(
        input_dim=INPUT_DIM, 
        num_classes=NUM_CLASSES, 
        bits=8, 
        dropout_rate=0.2
    )
    
    # Run the core inference engine
    # This simulates the on-device execution of the probabilistic quantized model
    mean_probs, epistemic_uncertainty = core_inference_engine(
        model=qpnn_model, 
        input_data=mock_input, 
        num_samples=100
    )
    
    # Determine final class
    final_classes = torch.argmax(mean_probs, dim=-1)
    
    print("--- Samsung R&D: QPNN Inference Results ---")
    for i in range(BATCH_SIZE):
        print(f"Sample {i}: Predicted Class {final_classes[i].item()} | "
              f"Confidence: {mean_probs[i][final_classes[i]].item():.4f} | "
              f"Uncertainty: {epistemic_uncertainty[i].item():.4f}")