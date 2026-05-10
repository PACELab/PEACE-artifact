# Mudi: Network Architecture-Based Interference Modeling Implementation Spec

## Problem & Scope

Implement the exact interference modeling methodology from the Mudi paper for GPU workload multiplexing. The system predicts interference between inference services and training tasks using ONLY network architecture characteristics explicitly mentioned in the paper. This implementation focuses on the core interference prediction mechanism without additional GPU profiling metrics.

**What we WILL implement:**
- Network layer counting for training tasks
- Piece-wise linear latency modeling for inference services
- Interference prediction using network architecture features
- Device selection based on interference estimates

**What we will NOT implement:**
- GPU kernel-level profiling
- NVIDIA DCGM metrics collection
- GPU utilization monitoring beyond what's needed for the core algorithm

## Sources & Anchors

**Paper:** "Multiplexing Dynamic Deep Learning Workloads with SLO-awareness in GPU Clusters"
- **Authors:** Wenyan Chen, Chengzhi Lu, Huanle Xu, Kejiang Ye, Chengzhong Xu
- **Venue:** EuroSys '25
- **Key Sections:**
  - Section 4.1.2 (p. 594-595): Interference modeling methodology
  - Figure 7 (p. 595): Network layer identification
  - Section 4.2 (p. 595): Online prediction process
  - Equation 1 (p. 594): Piece-wise linear function

## Data & Artifacts

**Available Workloads in Repository:**
- **ML Models:** albert-base-v2, bert-base-cased, mobilenet, mobilenet_v2, resnet-50, vit-base-patch16-224, vit_h_14, wav2vec2-base-960h, whisper-large-v2
- **CUDA Samples:** cudaTensorCoreGemm, fastWalshTransform, reductionMultiBlockCG, sortingNetworks, transpose

**Data Requirements:**
- Model architecture files (ONNX/TensorFlow for static graphs, PyTorch for dynamic)
- Latency measurements under different GPU% allocations (10%-90% in 10% increments)
- Batching sizes: [16, 32, 64, 128, 256, 512] as specified in paper

## Algorithm Plan

### 1. Network Architecture Feature Extraction
For each training task, extract the count of these layer types (as specified in Figure 7):
- `conv`: Convolutional layers
- `linear`: Linear layers
- `activations`: Activation functions (ReLU, etc.)
- `embeddings`: Embedding layers
- `encoder`: Encoder layers
- `decoder`: Decoder layers
- `flatten`: Flatten operations
- `batch_normalization`: Batch normalization layers
- `fc`: Fully connected layers
- `pooling`: Pooling layers
- `other_layers`: All remaining layer types

### 2. Piece-wise Linear Latency Modeling (Equation 1)
```
L^i_{b,Ψ} = {
    k^i_{Ψ,1} · (Δᵢ - Δ₀) + l₀,  if Δᵢ ≤ Δ₀
    k^i_{Ψ,2} · (Δᵢ - Δ₀) + l₀,  otherwise
}
```
Where:
- `L^i_{b,Ψ}`: Latency for inference service i with batching size b and co-located training task Ψ
- `Δᵢ`: GPU% allocation
- `k_{Ψ,1}, k_{Ψ,2}`: Slopes of piece-wise linear curve
- `(Δ₀, l₀)`: Cutoff point

### 3. Interference Prediction Model
**Input Features:** X = [Ψ, b]
- Ψ: Vector of layer counts [conv, linear, activations, embeddings, encoder, decoder, flatten, batch_norm, fc, pooling, other_layers]
- b: Batching size of inference service

**Output:** Y = [k^i_{Ψ,1}, k^i_{Ψ,2}, Δ₀, l₀]
- Parameters for piece-wise linear function

**Models:** Random Forest (RF), Support Vector Regression (SVR) as mentioned in paper

### 4. Device Selection Algorithm
1. For incoming training task j, extract network architecture Ψⱼ
2. For each inference service i, predict interference slopes using trained model
3. Calculate average slope across all batching sizes in {16, 32, 64, 128, 256, 512}
4. Assign training task to device with smallest average slope

## Critical Functions

### 1. `extract_network_architecture(model_path: str) -> Dict[str, int]`
**Purpose:** Extract layer counts from model architecture
**Input:** Path to model file (ONNX, TensorFlow, or PyTorch)
**Output:** Dictionary mapping layer types to counts
**Contract:**
```python
{
    "conv": int,           # Number of convolutional layers
    "linear": int,         # Number of linear layers
    "activations": int,    # Number of activation layers
    "embeddings": int,     # Number of embedding layers
    "encoder": int,        # Number of encoder layers
    "decoder": int,        # Number of decoder layers
    "flatten": int,        # Number of flatten operations
    "batch_normalization": int,  # Number of batch norm layers
    "fc": int,            # Number of fully connected layers
    "pooling": int,       # Number of pooling layers
    "other_layers": int   # All other layer types
}
```

### 2. `fit_piecewise_linear(gpu_percentages: List[float], latencies: List[float]) -> Tuple[float, float, float, float]`
**Purpose:** Fit piece-wise linear function to latency data
**Input:** GPU% values and corresponding latencies
**Output:** (k1, k2, delta0, l0) parameters
**Edge Cases:** Handle fewer than 6 samples, numerical instability

### 3. `train_interference_model(training_data: List[Tuple[Dict, int, Tuple]]) -> sklearn.BaseEstimator`
**Purpose:** Train interference prediction model
**Input:** List of (architecture_features, batch_size, piecewise_params) tuples
**Output:** Trained ML model (RF or SVR)

### 4. `predict_interference(model: sklearn.BaseEstimator, architecture: Dict[str, int], batch_size: int) -> Tuple[float, float, float, float]`
**Purpose:** Predict interference parameters for new training task
**Input:** Trained model, architecture features, batch size
**Output:** Predicted piece-wise linear parameters

### 5. `select_optimal_device(training_task_arch: Dict[str, int], inference_services: List[Dict]) -> int`
**Purpose:** Select best device for training task assignment
**Input:** Training task architecture, list of inference services with their models
**Output:** Device ID with minimal predicted interference

## Evaluation

### Metrics
- **Prediction Accuracy:** |y_pred - y_true| / y_true < 0.3 for all parameters
- **Device Selection Accuracy:** Match optimal device selection >90% of time
- **SLO Violation Rate:** < 2% for inference services

### Target Validation
- Reproduce Figure 11 accuracy results: average errors ≤ 0.23 for slopes
- Validate piece-wise linear fitting with R² > 0.85
- Test on unobserved workloads (last 4 training tasks in paper's Table 3)

## Risks & Unknowns

1. **Model Architecture Parsing:** Different frameworks may require different parsing approaches
2. **Layer Classification:** Ambiguity in classifying custom layers into the 11 categories
3. **Limited Training Data:** Only 11 layer types might not capture all interference patterns
4. **Dynamic vs Static Graphs:** PyTorch dynamic graphs require tracing for architecture extraction

## Acceptance Checklist

- [ ] Extract exact 11 layer types from all available workloads
- [ ] Implement piece-wise linear fitting with 6-sample requirement
- [ ] Train interference model achieving <0.3 prediction error
- [ ] Device selection matches optimal choice >90% of cases
- [ ] Integration test with available ML models and CUDA samples
- [ ] Validation on unseen workloads shows robust predictions
- [ ] No additional GPU profiling metrics beyond those explicitly mentioned
- [ ] Code follows repository structure (tests/mps/ directory)

## Implementation Notes

**Key Constraints:**
- Use ONLY the 11 network layer types specified in Figure 7
- Batch sizes must be from the set {16, 32, 64, 128, 256, 512}
- GPU% range: 10%-90% in 10% increments
- 6 training samples for piece-wise linear fitting (per paper's Table 2)
- Random Forest or SVR for interference prediction models