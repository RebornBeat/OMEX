OMEX: Omni-Execution Format

OMEX (Omni-Execution Format) is a universal, modular format and execution interface designed for distributed AI reasoning, scalable inference, and multi-device coordination. Built to seamlessly integrate with ZSEI (Zero-Shot Embedding Indexer), OMEX enables intelligent task execution across mobile, browser, Pi devices, servers, and specialized inference nodes — creating a unified AI operating environment.


---

Key Features

Modular Execution Containers: Encapsulate and structure AI task flows for reasoning, generation, analysis, or data routing.

Multi-Device Compatibility: Designed to execute and delegate tasks across diverse environments: Edge, Cloud, and Local.

ZSEI Integration: Leverages ZSEI for guidance generation, execution planning, and persistent state tracking.

Scalable Inference Coordination: Enables ultra-large models (e.g., 70B+) to execute on constrained devices by deferring memory-intensive computation to remote nodes.

Streaming & Chunk-Based Processing: Efficiently handles large content (code, video, documents) through adaptive chunking and stream execution.

Embeddable Agent API: Create agent logic that conforms to OMEX protocols, whether embedded in software, servers, or local hardware.



---

How It Works

OMEX works as both a format and runtime interpreter:

1. Prompt/Request: A user or system sends a request (text, code, task).


2. ZSEI Interpretation (optional): The request is routed through ZSEI which analyzes the intent, activates guidance frameworks, and returns an execution plan.


3. OMEX Format Generation: The plan is serialized as an OMEX container, composed of task nodes and execution graphs.


4. Distributed Execution:

On local devices (e.g., Pi, mobile) where compute is limited

On cloud/edge servers for large-model inference

Across federated agents for asynchronous task collaboration



5. Execution Completion & Feedback Loop: Results are streamed or returned through OMEX APIs, optionally logged into ZSEI for state retention or future recall.




---

OMEX Format Specification

An OMEX file is a structured, modular execution container.

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


---

Architecture Overview

OMEX Runtime

Interpreter: Parses and executes the OMEX format on supported hardware

Execution Dispatcher: Coordinates task distribution across nodes

ZSEI Agent API: Optional plugin to fetch guidance, embeddings, task plans

Memory Manager: Streams in/out data for memory-constrained systems


ZSEI Integration

ZSEI is used to:

Generate execution blueprints

Provide specialized guidance (e.g., code/document frameworks)

Handle long-running state

Act as a central coordinator for multi-task agents



---

Installation

Requirements

Rust 1.73+

Optional: Python 3.10+ (for glue layers or embedded scripts)

ONNX Runtime or custom inference backends

ZSEI installed (optional but recommended)


Install OMEX CLI

git clone https://github.com/your-org/omex.git
cd omex
cargo build --release
cargo install --path .


---

Quick Start

Run a Local OMEX Task

omex execute --file ./examples/code_summary.omex.json

Use with ZSEI

zsei process "Summarize my auth system" --output-format omex | omex execute


---

Example Use Cases

Distributed AI Agents: Use OMEX to create swarms of agents across phones, servers, and desktop clients.

Remote LLM Inference Control: Guide high-load inference operations from local devices.

Project-Wide Analysis: Schedule OMEX tasks to perform codebase audits, video captioning, or document generation across machines.

Federated Learning Coordination: Dispatch and retrieve local model updates using OMEX containers.



---

OMEX + ZSEI Agent Architecture

ZSEI Agents can:

Accept OMEX containers

Resolve embedded guidance tags (e.g., zsei://frameworks/code-update)

Execute per-node logic or route to capable devices

Report progress and logs back to the OMEX runtime


Agents can be installed as:

Desktop daemons

Server inference nodes

Mobile worker apps

WebAssembly (WASM) edge containers



---

Developer Tools

omex validate <file>: Validate format and structure

omex trace <task_id>: View execution status

omex convert: Convert between ZSEI tasks and OMEX

omex profile: Show available runtimes and device capacity



---

Configuration

Set global settings in ~/.omex/config.toml:

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


---

Roadmap

[x] OMEX Task Container Format

[x] OMEX CLI Runtime

[x] ZSEI API Integration

[ ] Agent Mesh Execution Support

[ ] WASM & Mobile Agent Launchers

[ ] Browser Plugin for OMEX Applets

[ ] Visual OMEX Designer GUI



---

License

MIT License. See LICENSE for details.


---

Learn More

ZSEI Documentation

OMEX Wiki

Join the discussion on Telegram

