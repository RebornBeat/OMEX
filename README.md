# OMEX: Omni-Execution Format

OMEX (Omni-Execution Format) is a universal, modular format and execution interface designed for AI model representation, distributed AI reasoning, scalable inference, and multi-device coordination. OMEX serves as both a model representation layer and an orchestration layer, enabling intelligent task execution across mobile, browser, Pi devices, servers, and specialized inference nodes — creating a unified AI operating environment. Unlike other formats such as ONNX, OMEX doesn't convert or work on top of existing models; it IS the model format itself, enabling even ultra-large models (70B+) to run directly on constrained devices.


---

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



---

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




---

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


---

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



---

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


---

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


---

## Example Use Cases

**Local Large Model Inference**: Run 70B+ parameter models on phones, Raspberry Pis, and laptops.

**Memory-Constrained AI**: Execute complex AI tasks on devices with limited RAM through smart chunking and memory management.

**Local-First AI Agents**: Create AI agents that operate entirely on local hardware without remote dependencies.

**Project-Wide Analysis**: Schedule OMEX tasks to perform codebase audits, video captioning, or document generation locally.

**Federated Learning Coordination**: Dispatch and retrieve local model updates using OMEX containers while keeping execution local.



---

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



---

## Developer Tools

**omex validate <file>**: Validate format and structure of OMEX model files

**omex trace <task_id>**: View execution status of running models

**omex convert**: Generate OMEX model representations from ZSEI tasks

**omex profile**: Show available runtimes and device capacity for model execution

**omex optimize**: Tune model parameters for specific device constraints

**omex layer-inspect**: Examine memory requirements of individual model layers



---

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
```


---

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



---

## License

MIT License. See LICENSE for details.


---

## Learn More

ZSEI Documentation

OMEX Wiki

Join the discussion on Telegram
