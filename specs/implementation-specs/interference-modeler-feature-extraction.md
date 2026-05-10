# Interference Modeler Feature Extraction

## Problem & Scope

Implement feature extraction methodology from the Mudi research paper for GPU workload interference modeling based on piece-wise linear latency functions. The system must extract workload characteristics to predict interference parameters (cutoff points, left/right slopes) that enable optimal co-location decisions for multiplexing inference services with training tasks on shared GPUs.

**Scope**: Feature extraction pipeline for interference prediction, not the full Mudi multiplexing system. Focus on extracting features from individual workloads to predict how they will interfere when co-located, using ground truth knee curve parameters as training targets.

## Sources & Anchors

**Paper**: "Multiplexing Dynamic Deep Learning Workloads with SLO-awareness in GPU Clusters" (EuroSys '25)
- **Section 4.1**: Piece-wise linear interference quantification (Equation 1)
- **Section 4.1.2**: Interference modeling using network architecture features
- **Figure 7**: Network layer feature extraction methodology
- **Section 5.1**: Optimization model connecting features to interference slopes
- **Section 7.3**: Accuracy evaluation of interference modeling

**Ground Truth Data**: `/tests/mps/eval_baselines/mudi/curvefit/fig/merged_freq1530/merged_knee.csv`
- Contains workload pairs with knee curve parameters: cutoff, value0, slope_left, slope_right
- Covers ML models: albert, bert, mobilenet, resnet, vit, wav2vec2, whisper
- Includes CUDA samples: cudaTensorCoreGemm, fastWalshTransform, reductionMultiBlockCG, etc.

## Data & Artifacts

### Input Data
- **Individual Workload Profiles**: GPU profiling data (nvidia-smi, dcgm metrics)
- **Model Architectures**: Network layer specifications for ML workloads
- **CUDA Sample Characteristics**: Kernel execution patterns for GPU samples
- **Ground Truth Labels**: Knee curve parameters from merged_knee.csv

### Feature Categories
1. **Network Architecture Features** (from Figure 7):
   - conv, linear, activations, embeddings, encoder, decoder
   - flatten, batch_normalization, fc, pooling, other_layers
2. **Resource Utilization Features**:
   - GPU memory usage patterns, SM utilization, memory bandwidth
3. **Execution Pattern Features**:
   - Kernel launch patterns, compute/memory ratio, synchronization overhead

### Synthetic Fixtures
- Minimal workload profiles for testing feature extraction pipeline
- Sample network architectures with known interference characteristics
- Unit test data for individual feature extractors

## Algorithm Plan

### 1. Workload Profiling Pipeline
```
For each workload w:
  1. Extract network architecture features → arch_features(w)
  2. Profile GPU resource usage → resource_features(w)
  3. Analyze execution patterns → execution_features(w)
  4. Combine into feature vector → X(w) = [arch, resource, execution]
```

### 2. Pairwise Interference Feature Generation
```
For each workload pair (w1, w2):
  1. Compute feature interactions → interaction_features(X(w1), X(w2))
  2. Apply workload-specific transformations → transform_features(w1, w2)
  3. Generate final feature vector → X_pair = [X(w1), X(w2), interactions]
```

### 3. Piece-wise Linear Parameter Prediction
From paper Equation 1: `L = k1*(Δ - Δ0) + l0` (left) or `k2*(Δ - Δ0) + l0` (right)

Target parameters from ground truth:
- **Cutoff point** (Δ0, l0): Resource allocation threshold
- **Left slope** (k1): Interference before cutoff
- **Right slope** (k2): Interference after cutoff

### 4. Multi-task Learning Framework
```
Input: X_pair(w1, w2)
Output: [k1, k2, Δ0, l0] for workload pair

Loss = α*MSE(k1) + β*MSE(k2) + γ*MSE(Δ0) + δ*MSE(l0)
```

### Key Formulas

**Network Architecture Encoding** (Section 4.1.2):
- `arch_vector = [n_conv, n_linear, n_activations, n_embeddings, n_encoder, n_decoder, n_flatten, n_batch_norm, n_fc, n_pooling, n_other]`

**Interference Score Calculation** (Section 5.2):
- `interference_score = mean(slopes_across_batches)`
- Used for device selection: assign training task to device with minimal interference

## Critical Functions

### 1. `extract_network_features(model_path: str) -> Dict[str, int]`
**Purpose**: Extract network architecture layer counts (Figure 7)
**Input**: Model file path (ONNX, PyTorch, etc.)
**Output**: Dictionary with layer type counts
**Edge Cases**: Dynamic graphs, unsupported layer types
**Test**: Verify against known model architectures (ResNet-50, BERT)

### 2. `extract_resource_features(workload_metrics: Dict) -> np.ndarray`
**Purpose**: Extract GPU resource utilization patterns
**Input**: GPU metrics (SM utilization, memory usage, bandwidth)
**Output**: Feature vector of resource characteristics
**Edge Cases**: Missing metrics, irregular sampling
**Test**: Validate with synthetic workload patterns

### 3. `compute_interference_features(w1_features: np.ndarray, w2_features: np.ndarray) -> np.ndarray`
**Purpose**: Generate pairwise interaction features
**Input**: Individual workload feature vectors
**Output**: Combined feature vector for interference prediction
**Edge Cases**: Feature dimension mismatches
**Test**: Ensure symmetric properties where applicable

### 4. `predict_knee_parameters(features: np.ndarray, models: Dict) -> Tuple[float, float, float, float]`
**Purpose**: Predict piece-wise linear parameters (k1, k2, Δ0, l0)
**Input**: Pairwise workload features, trained models
**Output**: Knee curve parameters
**Edge Cases**: Out-of-distribution workloads
**Test**: Compare predictions against ground truth within 15% error (Section 7.3)

### 5. `fit_piecewise_linear(gpu_percent: List[float], latency: List[float]) -> Dict`
**Purpose**: Fit piece-wise linear function to latency data
**Input**: GPU allocation percentages and corresponding latencies
**Output**: Fitted parameters and goodness-of-fit metrics
**Edge Cases**: Non-monotonic data, insufficient samples
**Test**: Synthetic piecewise functions with known parameters

## Evaluation

### Target Metrics (from Section 7.3)
- **Prediction Error**: |y_pred - y_true| / y_true < 0.15 (15% error threshold)
- **Parameter-specific targets**:
  - k1 (left slope): Mean error < 0.23
  - k2 (right slope): Mean error < 0.16
  - Δ0 (cutoff): Mean error < 0.05
  - l0 (value at cutoff): Mean error < 0.06

### Reproducible Results
- **Table**: Interference modeling accuracy per workload type
- **Figure**: Prediction error distribution across parameter types
- **Validation**: Cross-validation on unseen workload combinations

### Acceptance Criteria
1. Feature extraction completes for all workload types in ground truth data
2. Prediction accuracy meets paper benchmarks (Section 7.3)
3. Pipeline handles both ML models and CUDA samples
4. Feature vectors enable discrimination between high/low interference pairs

## Risks & Unknowns

### Technical Risks
- **Dynamic computation graphs**: PyTorch models may require runtime tracing
- **CUDA sample characterization**: Limited documentation for feature extraction
- **Feature scaling**: Different workload types may need normalization strategies
- **Temporal dependencies**: Some interference effects may depend on execution timing

### Implementation Assumptions
- **Model formats**: Assume access to model architecture definitions
- **Profiling data**: Assume consistent GPU metric collection methodology
- **Ground truth quality**: Assume knee curve fitting accuracy in provided data
- **Feature stability**: Assume extracted features remain consistent across runs

### Mitigation Strategies
- Implement fallback feature extraction for unsupported architectures
- Use robust fitting methods for noisy profiling data
- Cross-validate on multiple frequency settings beyond 1530MHz
- Implement feature importance analysis to identify critical predictors

## Acceptance Checklist

### Data Pipeline
- [ ] Successfully parse all workload types from ground truth CSV
- [ ] Extract features for all ML models (albert, bert, mobilenet, resnet, vit, wav2vec2, whisper)
- [ ] Extract features for all CUDA samples (cudaTensorCoreGemm, fastWalshTransform, etc.)
- [ ] Generate pairwise feature vectors for all workload combinations

### Feature Extraction
- [ ] Network architecture features match paper specifications (Figure 7)
- [ ] Resource utilization features capture GPU usage patterns
- [ ] Execution pattern features distinguish workload behaviors
- [ ] Feature vectors have consistent dimensionality across workloads

### Model Training & Prediction
- [ ] Multi-task learning framework predicts all four parameters (k1, k2, Δ0, l0)
- [ ] Prediction accuracy meets paper benchmarks for each parameter type
- [ ] Models handle unseen workload combinations (cross-validation accuracy > 80%)
- [ ] Interference ranking correlates with ground truth co-location performance

### Integration & Testing
- [ ] Unit tests cover all critical functions with edge cases
- [ ] Integration tests validate end-to-end feature extraction pipeline
- [ ] Performance tests ensure extraction completes within reasonable time bounds
- [ ] Documentation includes feature interpretation and usage guidelines

### Validation Against Ground Truth
- [ ] Reproduce paper's interference modeling accuracy results
- [ ] Validate on freq1530 dataset with known knee curve parameters
- [ ] Demonstrate feature importance aligns with domain knowledge
- [ ] Show extracted features enable effective co-location decisions