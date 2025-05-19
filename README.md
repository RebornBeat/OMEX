# OMEX: Omni-Execution Format

OMEX is a memory-optimized execution format for enabling large-scale inference locally on constrained devices — including mobile phones, Raspberry Pis, browsers, and lightweight servers.

Unlike traditional formats that assume large memory availability or require remote inference, OMEX is built for self-contained, locally executable task containers. It uses techniques like token window planning, sublayer routing, and smart data chunking to make models of any size feasible to run locally.

OMEX functions as a universal, modular format and execution interface designed for efficient AI reasoning, scalable inference, and multi-device coordination. Built to seamlessly integrate with ZSEI (Zero-Shot Embedding Indexer), OMEX enables intelligent task execution across mobile, browser, Pi devices, servers, and specialized inference nodes — creating a unified AI operating environment.

## Key Principles

- **Local First**: No distributed inference needed. All execution happens on the local device.
- **Memory Optimization Built-In**: Smart task chunking, layer targeting, quantized slices, and token stream planning.
- **ZSEI-Guided Task Planning (Optional)**: Use ZSEI to generate optimized execution flows and fetch guidance blueprints.
- **Modular Format**: Execution flows are broken into OMEX nodes (steps), each with clear input/output dependencies and memory profile hints.
- **Flexible Storage**: Some data/model slices can be stored locally, fetched, or cached from remote sources (not executed remotely).

## Key Features

- **Modular Execution Containers**: Encapsulate and structure AI task flows for reasoning, generation, analysis, or data routing.
- **Multi-Device Compatibility**: Designed to execute and delegate tasks across diverse environments: Edge, Cloud, and Local.
- **ZSEI Integration**: Leverages ZSEI for guidance generation, execution planning, and persistent state tracking.
- **Scalable Inference Coordination**: Enables ultra-large models (e.g., 70B+) to execute on constrained devices by deferring memory-intensive computation to remote nodes.
- **Streaming & Chunk-Based Processing**: Efficiently handles large content (code, video, documents) through adaptive chunking and stream execution.
- **Embeddable Agent API**: Create agent logic that conforms to OMEX protocols, whether embedded in software, servers, or local hardware.

## Use Cases

- Run SLMs or compressed LLMs on devices with 2–8GB RAM
- Execute long context reasoning with sliding token windows
- Process offline documents, codebases, or media using AI
- Build offline-capable agents that require no internet or cloud execution
- Support incremental inference over streamed data like chat logs, sensor data, or files
- Distributed AI Agents: Use OMEX to create swarms of agents across phones, servers, and desktop clients
- Remote LLM Inference Control: Guide high-load inference operations from local devices
- Project-Wide Analysis: Schedule OMEX tasks to perform codebase audits, video captioning, or document generation across machines
- Federated Learning Coordination: Dispatch and retrieve local model updates using OMEX containers

## How OMEX Works

OMEX works as both a format and runtime interpreter:

1. **Prompt/Request**: A user or system sends a request (text, code, task).

2. **ZSEI Interpretation (optional)**: The request is routed through ZSEI which analyzes the intent, activates guidance frameworks, and returns an execution plan.

3. **OMEX Format Generation**: The plan is serialized as an OMEX container, composed of task nodes and execution graphs.

4. **Execution Flow**:
   - **Plan**: Task is broken into minimal-memory nodes (manually or via ZSEI)
   - **Load**: Only required model weights or context windows are loaded at each step
   - **Infer**: Each step runs locally, caching intermediate outputs if needed
   - **Post-process**: Final result is decoded, saved, or passed to downstream steps

5. **Execution Completion & Feedback Loop**: Results are streamed or returned through OMEX APIs, optionally logged into ZSEI for state retention or future recall.

## Sample OMEX Container

```json
{
  "omex_version": "0.2.0",
  "task_id": "local-code-assist-432",
  "task_type": "code_generation",
  "context": {
    "execution_mode": "local",
    "memory_limit_mb": 2048,
    "zsei_guided": true,
    "device_profile": "mobile-mid-tier"
  },
  "execution_graph": {
    "entry_node": "node_parse",
    "nodes": [
      {
        "id": "node_parse",
        "type": "token_chunk_parse",
        "input": ["auth_handler.py"],
        "output": ["chunked_tokens.bin"],
        "runner": "local_cpu",
        "memory_profile": "low"
      },
      {
        "id": "node_infer",
        "type": "inference_slices",
        "depends_on": ["node_parse"],
        "input": ["chunked_tokens.bin"],
        "output": ["layered_logits.qpack"],
        "model_reference": "zsei://models/slm-code-v1?q8",
        "runner": "local_cpu",
        "memory_profile": "adaptive"
      },
      {
        "id": "node_finalize",
        "type": "output_merge_and_decode",
        "depends_on": ["node_infer"],
        "output": ["generated_code.py"]
      }
    ]
  },
  "storage_policy": {
    "allow_remote_fetch": false,
    "local_cache_dir": "./omex_cache",
    "model_prefetch": true
  },
  "execution_mode": "local",
  "memory_profile": {
    "max_ram_mb": 4096,
    "gpu_available": false
  },
  "model_requirements": {
    "model_family": "Mistral",
    "quantization": "int4",
    "max_sequence_tokens": 8192,
    "token_window_strategy": "sliding"
  },
  "task_flow": [
    {
      "id": "stage-0-load",
      "type": "model_load",
      "model_path": "local://models/mistral-int4.omex",
      "load_layers": [0, 10],
      "compression": "zstd",
      "output": "model_init"
    },
    {
      "id": "stage-1-preprocess",
      "type": "tokenize",
      "input": "user_input.txt",
      "output": "input_tokens"
    },
    {
      "id": "stage-2-infer",
      "type": "forward_pass",
      "model": "model_init",
      "input": "input_tokens",
      "token_window": {
        "window_size": 2048,
        "stride": 1024
      },
      "output": "logits"
    },
    {
      "id": "stage-3-generate",
      "type": "greedy_sample",
      "logits": "logits",
      "max_tokens": 256,
      "output": "generated_tokens"
    },
    {
      "id": "stage-4-postprocess",
      "type": "decode",
      "input": "generated_tokens",
      "output": "final_output"
    }
  ],
  "metadata": {
    "author": "BullPoster Labs",
    "date": "2025-05-15",
    "description": "Offline code assistant using mistral-int4"
  }
}
```

## OMEX Format Specification

An OMEX file is a structured, modular execution container.

```json
{
  "omex_version": "0.1.0",
  "task_id": "auth-system-analysis-456",
  "task_type": "code_analysis",
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
        "runner": "remote_inference_node"
      },
      {
        "id": "node_2",
        "type": "semantic_analysis",
        "depends_on": ["node_1"],
        "runner": "local_cpu",
        "guidance": "zsei://frameworks/code-analysis"
      },
      {
        "id": "node_3",
        "type": "summary_generation",
        "depends_on": ["node_2"],
        "runner": "zsei-guided-agent"
      }
    ]
  },
  "completion_policy": {
    "retry_on_fail": true,
    "max_runtime_minutes": 60
  }
}
```

## Architecture Overview

### OMEX Runtime

- **Interpreter**: Parses and executes the OMEX format on supported hardware
- **Execution Dispatcher**: Coordinates task distribution across nodes
- **ZSEI Agent API**: Optional plugin to fetch guidance, embeddings, task plans
- **Memory Manager**: Streams in/out data for memory-constrained systems

### OMEX Execution Engine

Any compatible OMEX runtime or lightweight interpreter can:
- Parse the task flow
- Load models in segments or quantized format
- Respect memory profiles and layer prefetching
- Handle token stream shifts efficiently
- Execute each task step in order with cached state reuse

The OMEX execution engine should be embeddable in:
- Python (via a minimal runtime)
- Rust/C++ (native clients or edge runtimes)
- WASM (browser/embedded support)

### ZSEI Integration

ZSEI is used to:

- Generate execution blueprints
- Provide specialized guidance (e.g., code/document frameworks)
- Handle long-running state
- Act as a central coordinator for multi-task agents

ZSEI enhances OMEX with:
- Execution guidance: Suggests how to break a task into memory-efficient nodes
- Model references: Supplies compressed model URLs or blueprints (Q4, Q8, LoRA layers)
- Pre-planning: Allows a one-time remote setup before full offline execution

Example command:
```bash
zsei plan "Summarize this PDF" --target-device pi4 --output omex.json
omex execute --file omex.json
```

OMEX can run entirely without ZSEI, but gains efficiency when ZSEI is used as a planning or guidance layer.

## Device Compatibility

- **Phones/Tablets**: Android, iOS (via WASM or native)
- **Raspberry Pi**: v4, v5
- **Browsers**: Through WASM bindings (coming soon)
- **Laptops/Desktops**: Lightweight CPU or GPU execution

## Example Use Cases

| Scenario | Model Size | Device | Memory Used |
|----------|------------|--------|-------------|
| PDF summarizer (offline) | 7B Q8 | Pi 5 | ~1.2GB |
| Chatbot with memory rewind | 3B LoRA | Android Phone | ~800MB |
| Code explainer (split load) | 13B Q4 | Laptop | ~2GB |

## Installation

### Requirements

- Rust 1.73+
- Optional: Python 3.10+ (for glue layers or embedded scripts)
- ONNX Runtime or custom inference backends
- ZSEI installed (optional but recommended)

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

## OMEX + ZSEI Agent Architecture

ZSEI Agents can:

- Accept OMEX containers
- Resolve embedded guidance tags (e.g., zsei://frameworks/code-update)
- Execute per-node logic or route to capable devices
- Report progress and logs back to the OMEX runtime

Agents can be installed as:

- Desktop daemons
- Server inference nodes
- Mobile worker apps
- WebAssembly (WASM) edge containers

## Developer Tools

- `omex validate <file>`: Validate format and structure
- `omex trace <task_id>`: View execution status
- `omex convert`: Convert between ZSEI tasks and OMEX
- `omex profile`: Show available runtimes and device capacity
- `omex execute --file task.omex.json`: Runs a local OMEX file
- `omex validate task.omex.json`: Validates syntax and structure
- `omex trace --task-id`: Views node-by-node logs
- `omex debug --profile device`: Estimates memory usage for each node

## Configuration

Set global settings in `~/.omex/config.toml`:

```toml
[runtime]
default_runner = "auto"
allow_remote = true
enable_zsei_integration = true

[storage]
cache_dir = "~/.omex/cache"
log_retention_days = 7

[zsei]
host = "http://localhost:8801"
agent_token = "YOUR_ZSEI_API_TOKEN"
```

## Why OMEX?

- **Works anywhere**: You don't need a server or cloud GPU.
- **Highly customizable**: Run only part of a model, use fallback strategies, or run in hybrid (online/offline) modes.
- **Secure and private**: All inference is done on-device, no data leaves the user's machine.
- **Pluggable**: Swap in different model families or task flows easily.

## Roadmap

- [x] OMEX Task Container Format
- [x] OMEX CLI Runtime
- [x] ZSEI API Integration
- [ ] Agent Mesh Execution Support
- [ ] OMEX-compatible quantizer tools
- [ ] WASM & Mobile Agent Launchers
- [ ] Browser Plugin for OMEX Applets
- [ ] Visual OMEX Designer GUI
- [ ] Add support for streaming output and dynamic token re-entry
- [ ] WASM + Mobile SDKs
- [ ] Auto-generated OMEX from prompts via ZSEI compiler
- [ ] Model caching & sharing via OMEX-Hub
- [ ] Multi-modal container support (audio, vision)

## Coming Soon

- WASM-native runner for browsers
- Visual OMEX graph editor
- OMEX-compatible quantizer tools
- Prebuilt SLM packs optimized for OMEX
- Token streaming APIs for chat agents

## License

OMEX is licensed under the MIT License. See LICENSE for details.

## Learn More

- [OMEX Wiki](https://github.com/your-org/omex/wiki)
- [ZSEI Documentation](https://zsei.xyz)
- Join the discussion on [Telegram](https://t.me/zsei_community)
