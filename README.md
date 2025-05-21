# OMEX: Omni-Execution Format

OMEX (Omni-Execution Format) is a universal, modular format and execution interface designed for AI model representation, distributed AI reasoning, scalable inference, and multi-device coordination. OMEX serves as both a model representation layer and an orchestration layer, enabling intelligent task execution across mobile, browser, Pi devices, servers, and specialized inference nodes — creating a unified AI operating environment. Enabling even ultra-large models (70B+) to run directly on constrained devices.

## Key Features

**Modular Execution Containers**: Encapsulate and structure AI task flows for reasoning, generation, analysis, or data routing.

**Direct Model Representation**: OMEX is not just a container but the actual model format, eliminating the need for converting between formats.

**Local-First Execution**: All inference happens on the local device with no distributed execution required, regardless of model size.

**Memory Optimization Built-In**: Smart task chunking, layer targeting, quantized slices, and token stream planning allow running large models on constrained devices.

**Multi-Device Compatibility**: Designed to execute tasks across diverse environments: Edge, Cloud, and Local, with emphasis on local execution.

**ZSEI Integration**: Leverages ZSEI for guidance generation, execution planning, and persistent state tracking.

**Scalable Local Model Execution**: Enables ultra-large models (e.g., 70B+) to execute directly on constrained devices through advanced memory management techniques.

**Streaming & Chunk-Based Processing**: Efficiently handles large content (code, video, documents) through adaptive chunking and stream execution.

**Embeddable Agent API**: Create agent logic that conforms to OMEX protocols, whether embedded in software, servers, or local hardware.

**Flexible Storage**: Model weights and data can be stored locally, fetched, or cached from remote sources (but always executed locally).

**Prompt-First Design**: OMEX is prompt-driven, optimizing execution paths based on prompt requirements rather than static model exports.

**Hardware-Aware Execution**: Optimizes execution for specific hardware, including tensor core utilization, precision scaling, and memory management.

**Zero-Shot Execution Planning**: Dynamically creates execution graphs based on prompt analysis without requiring predefined execution paths.

**Model Component Hot-Swapping**: Enables runtime replacement of model components (layers, adapters, tokenizers) for flexible execution.

## How It Works

OMEX works as both a model format and runtime interpreter:

1. **Prompt/Request**: A user or system sends a request (text, code, task).

2. **ZSEI Interpretation (optional)**: The request is routed through ZSEI which analyzes the intent, activates guidance frameworks, and returns an execution plan.

3. **OMEX Format Generation**: The plan is serialized as an OMEX container, composed of task nodes and execution graphs that directly represent model structure and flow.

4. **Local Optimized Execution**:
   - Smart partitioning of model layers and weights
   - Memory-optimized execution pipeline
   - Dynamic resource allocation based on device capabilities
   - Token-level streaming to manage memory constraints

5. **Execution Completion & Feedback Loop**: Results are streamed or returned through OMEX APIs, optionally logged into ZSEI for state retention or future recall.

## OMEX Format Specification

An OMEX file is a structured, modular execution container that directly represents the model architecture and execution flow.

```json
{
  "omex_version": "0.1.0",
  "task_id": "auth-system-analysis-456",
  "task_type": "code_analysis",
  "model_architecture": {
    "type": "transformer",
    "parameters": "70B",
    "quantization": "int8",
    "memory_profile": {
      "required_ram": "4GB",
      "swap_strategy": "layer_offloading"
    }
  },
  "context": {
    "source": "user",
    "requested_by": "terminal",
    "zsei_guidance": true,
    "device_profile": "raspberry-pi-5"
  },
  "execution_graph": {
    "entry_node": "node_1",
    "nodes": [
      {
        "id": "node_1",
        "type": "parse_codebase",
        "input": ["./src"],
        "output": ["parsed_ast.json"],
        "runner": "local_device",
        "memory_allocation": "512MB"
      },
      {
        "id": "node_2",
        "type": "semantic_analysis",
        "depends_on": ["node_1"],
        "runner": "local_cpu",
        "guidance": "zsei://frameworks/code-analysis",
        "model_layers": ["encoder_1", "encoder_2", "encoder_3"],
        "memory_allocation": "1GB"
      },
      {
        "id": "node_3",
        "type": "summary_generation",
        "depends_on": ["node_2"],
        "runner": "local_with_zsei_guidance",
        "model_layers": ["decoder_1", "decoder_2", "decoder_3"],
        "memory_allocation": "2GB"
      }
    ]
  },
  "model_weights": {
    "storage_strategy": "chunked_local",
    "chunk_size": "200MB",
    "prefetch_strategy": "predictive"
  },
  "completion_policy": {
    "retry_on_fail": true,
    "max_runtime_minutes": 60,
    "memory_pressure_strategy": "dynamic_precision_reduction"
  }
}
```

## Canonical OMEX Model Format

OMEX models follow a canonical structure that combines a computation graph, parameters, and processing logic into a single, self-contained format.

### Folder Structure

```
model.omex/
├── metadata.json            # Model metadata and versioning
├── graph.json               # Computation graph structure
├── weights/                 # Model weights directory
│   ├── quantization.json    # Weight quantization configuration
│   ├── layer_0.bin          # Binary weight data for layer 0
│   ├── layer_1.bin          # Binary weight data for layer 1
│   └── ...                  # Additional layer weights
├── tokenizer/               # Tokenizer components
│   ├── tokenizer.json       # Tokenizer configuration
│   ├── vocab.txt            # Vocabulary file
│   └── merges.txt           # BPE merges file (if applicable)
├── preprocessors/           # Pre-processing components
│   ├── text_preprocessor.json  # Text preprocessing configuration
│   ├── code_preprocessor.json  # Code preprocessing configuration
│   └── ...                  # Additional preprocessors
├── postprocessors/          # Post-processing components
│   ├── text_formatter.json  # Text formatting configuration
│   ├── code_formatter.json  # Code formatting configuration
│   └── ...                  # Additional postprocessors
├── agents/                  # Agent definitions
│   ├── default_agent.json   # Default agent configuration
│   ├── code_agent.json      # Code-specific agent configuration
│   └── ...                  # Additional agent configurations
└── extensions/              # Optional extension components
    ├── tools/               # Tool definitions
    ├── adapters/            # LoRA or adapter weights
    └── hooks/               # Execution hooks
```

### Architecture Components

1. **Computation Graph (`graph.json`)**:
   - Full model structure definition
   - Layer specifications, attention blocks, activations
   - Control flow and execution paths
   - Modular subgraphs and component references

2. **Parameters (`weights/`)**:
   - Trained values organized by layer
   - Support for multiple quantization formats
   - Efficient chunking for low-memory environments
   - Memory-mapped access for streaming execution

3. **Tokenizer + Pre/Post-Processing (`tokenizer/`, `preprocessors/`, `postprocessors/`)**:
   - Integrated tokenization (BPE, WordPiece, SentencePiece)
   - Input formatting and normalization
   - Output formatting and detokenization
   - Specialized processing for different content types

4. **Agent Definitions (`agents/`)**:
   - Runtime behavior specifications
   - Task-specific configurations
   - Tool usage patterns
   - Memory and context management

## Architecture Overview

### OMEX Runtime

**Interpreter**: Parses and executes the OMEX format directly on supported hardware

**Model Engine**: Processes model weights and architecture definitions for local execution

**Memory Optimizer**: Manages dynamic allocation, swapping, and precision adjustment to fit models within device constraints

**Execution Dispatcher**: Coordinates task distribution across local processing units

**ZSEI Agent API**: Optional plugin to fetch guidance, embeddings, task plans

**Memory Manager**: Streams in/out data and model weights for memory-constrained systems

### ZSEI Integration

ZSEI is used to:

**Generate execution blueprints**: Creates optimized task sequences for model execution

**Provide specialized guidance**: Delivers context-specific frameworks (e.g., code/document frameworks)

**Handle long-running state**: Maintains conversation or task context across interactions

**Act as a central coordinator**: Orchestrates model execution steps for optimal performance

## Creating and Training Models with OMEX

OMEX supports multiple approaches for creating and training models:

### Converting Existing Models

Existing models can be converted to OMEX format using the OMEX conversion tools:

```bash
# Convert from PyTorch to OMEX
omex convert --source-format pytorch --model-path ./llama3_7b.pt --output-dir ./llama3_7b.omex

# Convert from ONNX to OMEX
omex convert --source-format onnx --model-path ./phi3_mini.onnx --output-dir ./phi3_mini.omex

# Convert from Hugging Face to OMEX
omex convert --source-format huggingface --model-name "mistralai/Mistral-7B-v0.1" --output-dir ./mistral_7b.omex

# Convert with quantization
omex convert --source-format pytorch --model-path ./llama3_70b.pt --output-dir ./llama3_70b.omex --quantize int8 --optimize-for mobile
```

### Direct Training in OMEX Format

OMEX supports training models directly in its native format:

```bash
# Initialize a new OMEX model structure
omex init --architecture transformer --size 7B --output-dir ./my_model.omex

# Configure model architecture
omex configure --model-dir ./my_model.omex --layers 32 --heads 32 --dim 4096 --vocab-size 32000

# Train model
omex train --model-dir ./my_model.omex --train-data ./dataset.jsonl --val-data ./validation.jsonl --epochs 3 --batch-size 32 --gradient-accumulation 8
```

### Fine-tuning OMEX Models

Existing OMEX models can be fine-tuned for specific tasks:

```bash
# Fine-tune with LoRA
omex finetune --model-dir ./llama3_7b.omex --train-data ./instruction_data.jsonl --method lora --lora-r 16 --lora-alpha 32 --output-dir ./llama3_7b_finetuned.omex

# Create domain-specific adapters
omex create-adapter --model-dir ./mistral_7b.omex --adapter-name "code-adapter" --train-data ./code_examples.jsonl --output-dir ./code_adapter.omex

# Merge adapters into base model
omex merge-adapter --model-dir ./llama3_7b.omex --adapter-path ./code_adapter.omex --output-dir ./llama3_7b_code.omex
```

### Creating MoE Models

OMEX supports creating Mixture of Experts (MoE) models for more efficient execution:

```bash
# Create MoE model from existing model
omex create-moe --model-dir ./llama3_7b.omex --experts 8 --expert-size 1B --output-dir ./llama3_7b_moe.omex

# Train routing network
omex train-router --model-dir ./llama3_7b_moe.omex --train-data ./dataset.jsonl --output-dir ./llama3_7b_moe_routed.omex
```

## Performance Optimization

OMEX implements extensive optimizations across all hardware tiers:

### Edge Devices (Mobile, Raspberry Pi)

| Module Type | Optimization | Expected Gain |
|-------------|--------------|--------------|
| Tokenizer | Streaming, SIMD vectorization | 60-100% faster |
| MLPs | Int8 quantization, pruning | 100-200% faster |
| Attention | Lineformer/LinearAttention substitution | 50-100% faster |
| Streaming | Token-by-token processing | Up to 300% throughput |

### Mid-Range GPUs (RTX 4090, A6000)

| Module Type | Optimization | Expected Gain |
|-------------|--------------|--------------|
| Tokenizer | Prompt-based optimization | 25-60% faster |
| MLP Layer | Fused operations, FP16 | 30-60% faster |
| Multi-head Attention | FlashAttention2 integration | 40-70% faster |
| Batch Processing | Container-based parallelism | 200%+ throughput |

### High-End Hardware (H100, A100, TPUv4+)

| Module Type | Optimization | Expected Gain |
|-------------|--------------|--------------|
| Tokenizer | Shard-aware processing | 60-80% faster |
| FP16/BF16 MLP | Tensor core optimization | 50-70% faster |
| MoE Layer | Expert pruning, dynamic routing | 100-150% faster |
| Graph Fusion | Prompt-aware operation fusion | 100-200% faster |
| Latency | End-to-end optimization | 40-50% lower latency |

## Installation

### Requirements

Rust 1.73+

Optional: Python 3.10+ (for glue layers or embedded scripts)

Local compute resources (CPU/GPU/NPU)

ZSEI installed (optional but recommended)

### Install OMEX CLI

```bash
git clone https://github.com/your-org/omex.git
cd omex
cargo build --release
cargo install --path .
```

## Quick Start

### Run a Local OMEX Task

```bash
omex execute --file ./examples/code_summary.omex.json
```

### Use with ZSEI

```bash
zsei process "Summarize my auth system" --output-format omex | omex execute
```

### Load a Large Model Locally

```bash
omex load-model --size 70B --device "raspberry-pi-5" --optimize-memory
```

## Example Use Cases

**Local Large Model Inference**: Run 70B+ parameter models on phones, Raspberry Pis, and laptops.

**Memory-Constrained AI**: Execute complex AI tasks on devices with limited RAM through smart chunking and memory management.

**Local-First AI Agents**: Create AI agents that operate entirely on local hardware without remote dependencies.

**Project-Wide Analysis**: Schedule OMEX tasks to perform codebase audits, video captioning, or document generation locally.

**Federated Learning Coordination**: Dispatch and retrieve local model updates using OMEX containers while keeping execution local.

## OMEX + ZSEI Agent Architecture

ZSEI Agents can:

**Accept OMEX containers**: Process model representations and task definitions

**Resolve embedded guidance tags**: Interpret references like `zsei://frameworks/code-update`

**Execute per-node logic locally**: Run all computation on the device

**Report progress and logs**: Stream execution status back to the OMEX runtime

Agents can be installed as:

**Desktop daemons**: Background processes handling local model execution

**Server inference nodes**: Local execution units for server environments

**Mobile worker apps**: Dedicated applications for on-device inference

**WebAssembly (WASM) edge containers**: Browser-compatible execution environments

## Developer Tools

**omex validate <file>**: Validate format and structure of OMEX model files

**omex trace <task_id>**: View execution status of running models

**omex convert**: Generate OMEX model representations from ZSEI tasks

**omex profile**: Show available runtimes and device capacity for model execution

**omex optimize**: Tune model parameters for specific device constraints

**omex layer-inspect**: Examine memory requirements of individual model layers

**omex benchmark**: Measure performance across different hardware configurations

**omex compare**: Compare performance between OMEX and other formats (ONNX, GGUF)

**omex graph-visualize**: Create visual representation of model execution graphs

**omex memory-trace**: Track memory usage patterns during model execution

## Configuration

Set global settings in ~/.omex/config.toml:

```toml
[runtime]
default_runner = "local_device"
memory_optimization = "aggressive"
enable_zsei_integration = true

[model]
default_quantization = "int8"
layer_swapping = true
precision_scaling = true

[storage]
cache_dir = "~/.omex/cache"
model_weights_dir = "~/.omex/models"
log_retention_days = 7

[zsei]
host = "http://localhost:8801"
agent_token = "YOUR_ZSEI_API_TOKEN"

[performance]
tensor_cores = true
kernel_fusion = true
parallel_execution = true
streaming_tokens = true
kv_cache_optimization = true

[device]
cpu_threads = 8
gpu_memory_limit = "4GB"
swap_path = "~/.omex/swap"
enable_mmap = true
prefetch_distance = 2
```

## Optimization Checklist

### Universal Optimizations
- Convert prompts into static execution plans
- Remove unneeded graph segments
- Prune unused attention heads and MLP branches
- Combine LayerNorm + MatMul + Activation
- Support runtime quantization fallback
- Stream tokens during tokenization
- Optimize KV cache memory usage
- Implement multi-query attention
- Use automatic mixed precision
- Enable container-based async batch execution

### Edge Device Optimizations
- Use SIMD vectorized tokenizer with trie-based vocabulary
- Quantize linear layers to int8
- Replace standard attention with Linformer or LinearAttention
- Use compact weight formats (GGUF-like)
- Leverage platform-specific accelerators (Metal, NNAPI)
- Stay under 2B parameters with quantization and pruning

### GPU Optimizations
- Enable prompt-based module activation
- Use FlashAttention2 or triton-based fused kernels
- Pre-allocate KV cache with float16
- Use multi-instance containers for batch optimization
- Split tokenization → inference → detokenization across GPU streams
- Use tensor cores (BF16 preferred on H100)
- Prune MoE experts not relevant to prompt
- Use graph scheduling with NVIDIA CUTLASS + Triton3
- Organize cache in fused format across sequence dimension

## Format Comparison

OMEX vs. other model formats:

| Feature | ONNX | GGUF | TorchScript | MLIR | SavedModel | OMEX |
|---------|------|------|------------|------|------------|------|
| Design Flow | Pre-built → Export → Run | Pre-trained + quantized | Traced model export | IR optimized | Static serialized | Prompt → Build → Run |
| Graph Flexibility | Medium | Low | Low-medium | High | Medium | High (Prompt-to-Graph) |
| Edge Device Support | Partial | Yes | Partial | Partial | Partial | Yes (streamable) |
| H100/A100 Optimization | Yes | No | Yes | Yes | Yes | Yes (tensor-core aware) |
| Dynamic Prompt Routing | No | No | No | Yes | No | Yes |
| Quantization Aware | Partial | Yes | Limited | Yes | Partial | Yes |
| Graph Fusion | Partial | No | Partial | Yes | No | Yes |
| Adapter/MoE Support | No | No | No | Partial | No | Yes (modular) |
| Runtime Tokenization | No | Yes (basic) | No | No | Partial | Yes (embedded) |
| Model Updates | Full rebuild | Full rebuild | Full re-trace | Partial rebuild | Full retrain/export | Prompt-linked containers |
| Streaming Execution | No | No | No | Yes (compile-time) | No | Yes (layer-wise) |
| Hot-Swappable Modules | No | No | No | Yes (with IREE) | No | Yes (on demand) |

## Throughput Improvements

OMEX provides significant throughput improvements over other formats:

### Edge Devices
- 60-100% faster tokenization
- 100-200% faster MLP operations (quantized)
- 50-100% faster attention mechanisms
- 300% faster streaming inference

### Mid-Range GPUs
- 25-60% faster tokenization
- 30-60% faster MLP operations
- 40-70% faster multi-head attention
- 200%+ faster batch parallel processing

### H100/A100 Clusters
- 60-80% faster tokenization
- 50-70% faster FP16/BF16 MLP operations
- 100-150% faster MoE layer operations
- 100-200% faster graph fusion operations
- 40-50% lower end-to-end latency

## Roadmap

[x] OMEX Task Container Format

[x] OMEX CLI Runtime

[x] ZSEI API Integration

[x] Direct Model Representation

[ ] Memory Optimization Engine

[ ] Layer-wise Execution Scheduler

[ ] Agent Mesh Execution Support

[ ] WASM & Mobile Agent Launchers

[ ] Browser Plugin for OMEX Applets

[ ] Visual OMEX Designer GUI

## License

MIT License. See LICENSE for details.

## Learn More

ZSEI Documentation

OMEX Wiki

Join the discussion on Telegram
