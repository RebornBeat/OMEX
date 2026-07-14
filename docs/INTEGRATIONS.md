# OMEX in Host Systems

OMEX is **standalone by design**: any application can embed the parser, linker, and realizer directly. But its full power appears inside an orchestrated system — one that turns graphs into intent, executes tasks, and needs responses rendered back. This document explains, for readers who have never heard of them, the two reference systems OMEX was designed alongside — **Ozone-Studio** and **ZSEI** — and, generically, what *any* host must provide.

## 1. What a Host/Orchestrator Is

An orchestrator (sometimes "AGI harness") is a system that receives a user's prompt, decomposes it into structure and intent, plans and executes tasks across specialized pipelines, and delivers a response. In such systems, OMEX models are **executors**: they replace the general-LLM calls that previously did structural work, at a fraction of the cost, with identical graph contracts.

## 2. Primer: Ozone-Studio (the reference orchestrator)

Ozone-Studio is a multi-modality orchestration platform. Text, code, images, audio, math, and many other modalities each have a **pipeline** that analyzes content and produces a **structural graph** for it. A central orchestrator coordinates: it normalizes the prompt, builds an intent structure, matches or creates execution plans, runs steps across pipelines, and delivers the response.

**The text pipeline as a graph factory.** Long inputs are chunked **once, on intake** — chunks exist only to fit any model's context window, enable parallel processing, and anchor byte positions; after nodes are created, all subsequent work is graph-based. The pipeline runs one of two host-side processing pathways (Ozone-Studio calls them Path 1 and Path 2 internally — these are **host concepts, not OMEX concepts**):

- **The whole-chunk pathway (deconstructor, top-down):** a capable model takes the full chunk and emits its structure and corrections in one sweep — sections, paragraphs, sentences, grammar, embedded non-prose spans.
- **The granular pathway (constructor, bottom-up):** designed for constrained models (e.g., 4K-context SLMs). Sentences are identified **one at a time, in order**, each grammar-corrected, each confirmed by repeated independent yes/no validation (five consecutive confirmations) before becoming a permanent node; section markers and embedded modality spans are detected the same way, one at a time, in order; paragraphs and higher structure are then **constructed afterward from the sentence graph** by evidence traversal. Constant-size prompt state per step prevents context explosion.

**Why this matters to OMEX:** the granular pathway's validation discipline is the **data-QA gate** that manufactures OMEX's training corpora; and the whole-chunk pathway is the shape of **OMEX-native operation** — a trained parser emits an entire chunk's graph in one non-autoregressive pass, collapsing every granular loop into a single call plus cheap verification. After all chunks are processed, grammar relationships and pairwise cross-sentence analysis run **graph-natively over the assembled corpus** — exactly the structures OMEX trains on and emits.

**The intent structure (AMT) — how the host actually "thinks."** Ozone-Studio builds an *Abstract Meaning Tree* — a tree of intents, branches, and details — never by asking a model "what is the intent?", but by a five-stage deterministic process over the graphs:

1. **Context pool accumulation (Stage 1).** Starting from sentence nodes, walk typed grammar edges, coreference chains, and cross-sentence relationships, accumulating pools of evidence that belong together — evidence expansion, not similarity clustering.
2. **Neighborhood continuity evaluation (Stage 2).** Adjacent pools merge or separate based on *evidence of continuity* — shared entities, dependency links, discourse relations, ownership boundaries — which is what keeps distinct projects distinct even when they share vocabulary.
3. **Promotion (Stage 3).** Sufficient evidence promotes a neighborhood to an **intent node** carrying its summary, actor/action/target/constraints, full supporting evidence, and — because OMEX graphs carry knowledge references on their sentence nodes, resolved and pool-bounded per docs/KNOWLEDGE.md — its **Knowledge Context Pool, already aggregated**. No separate knowledge pass exists anywhere in the tree.
4. **Task creation.** Concrete tasks derive from branches, each linked back to its evidence; dependencies, loops, and sub-steps are explicit.
5. **Response Graph assembly at checkpoints.** When a branch needs a user-facing answer, its pool and verified results assemble deterministically into a Response Graph (docs/RESPONSE_GRAPH.md) and go to the rendering ladder.

The AMT's internal thinking generates **no text** — reasoning is traversal, inspectable node by node, with every conclusion tracing to specific grammar evidence and fabric edges (docs/EXECUTION.md §2 for the generic form of this process).

**Response generation.** When a task or checkpoint needs a user-facing answer, the orchestrator assembles the Response Graph from that branch's verified results and hands it to the rendering ladder — the OMEX Realizer once trained, the validated stand-ins before then (docs/REALIZER.md §6) — the reverse-traversal loop from docs/EXECUTION.md, embodied.

## 3. Primer: ZSEI (the reference knowledge fabric)

ZSEI (Zero-Shot Embedded Indexer) is **not a database** — it is a semantic knowledge fabric: it stores meaning, relationships, and traversable structure. Everything in it is a **container** (concepts, topics, methodologies, execution blueprints, modality graphs, external references), organized in a logical hierarchy (e.g., `/Modality/Text/...`, knowledge domains, workspaces) and connected by typed relationships.

Key properties, briefly:

- **Hybrid storage.** A memory-mapped global index with fixed-size headers gives O(1) structural hops (parent/child/version) at billion-container scale; rich per-container JSON holds metadata, keywords, topics, embeddings, and relationships; only hot containers stay in RAM.
- **Three traversal modes, combined.** *Structural* (walk the hierarchy), *semantic* (embedding similarity), *contextual* (follow typed relationship edges) — each covers the others' blind spots; combined search merges and ranks all three.
- **Zero-shot verification.** Top candidates from any search can be verified by a model asking, in effect, "is this actually relevant?" — a quality filter requiring no training, cached to avoid repeat cost.
- **Link, don't copy.** Content is referenced (files by path+hash, URLs with semantic snapshots, packages by registry coordinates) — never duplicated.

**What OMEX uses ZSEI for:** (a) storing the training graph corpora; (b) being the address space the **KnowledgeLinker** resolves into — an HNSW index over container embeddings, batched ANN per chunk, zero-shot verification on low-margin links; (c) providing the **typed, confidence-scored relationship edges** that bounded knowledge traversal walks to build Knowledge Context Pools (docs/KNOWLEDGE.md §3); (d) storing the modality graphs OMEX models emit at runtime, where semantic hooks then enrich them with inferred relationship edges.

## 4. What *Any* Host Must Provide

OMEX does not require Ozone-Studio or ZSEI. Substitutes must provide:

| Requirement | For | Reference implementation |
|---|---|---|
| A graph-generation/validation pipeline (or an existing schema-conformant corpus) | Training | Ozone-Studio text pipeline, granular pathway |
| A knowledge fabric with **addressable paths** and **entry embeddings** | KnowledgeLinker resolution | ZSEI |
| **Typed relationship edges with confidences** on fabric entries | Bounded knowledge traversal (Knowledge Context Pools) | ZSEI contextual edges |
| An intent/meaning layer that consumes grammar graphs | Understanding | AMT traversal |
| A **Response Graph assembler** (skeleton + pre-filled verified content + provenance + voice) | Generation handoff | Ozone-Studio orchestrator (docs/RESPONSE_GRAPH.md §4) |
| A rendering ladder floor (template renderer at minimum) | Day-zero responses | Tier 0 (docs/REALIZER.md §6) |

## 5. Memory Strategy in an Integrated Deployment

The generalized tiering — always-resident weights and indexes, on-demand fabric records and pool neighborhoods with prefetch-on-detection, write-behind for emitted graphs and flywheel pairs — is specified once, host-agnostically, in **docs/DEPLOYMENT.md §2–§3**, and applies to ZSEI-backed deployments unchanged. The ZSEI-specific notes: the global mmap rides the OS page cache; the linker's HNSW index over container embeddings is the latency-critical always-resident item; resolution caches key on normalized span hashes with TTL invalidation on container updates; and fabric traversal is pointer-chasing CPU work that runs as the second lane against GPU encoder passes, joined at chunk boundaries — neither ever waits on the other.

## 6. Application Pattern: Knowledge-Genome Ingestion (Evidence Graphs)

A worked example of OMEX feeding a self-organizing knowledge system. Suppose a platform ingests thousands of courses to build an ontology of human knowledge. The mature approach does **not** classify by asking semantic questions ("Is Heat Treatment a Topic?") — perspective-dependent questions produce unstable answers. Instead it classifies by **deterministic metrics over observed graph evidence**: reuse scores (how many contexts reference a node), atomicity (can it split), cross-domain spread, internal cohesion, dependency depth, canonical confidence — computed by traversal, feeding **promotion/demotion rules** so the ontology self-organizes as evidence accumulates (a node like *Digital Twin* gets promoted once it demonstrably recurs across manufacturing, robotics, healthcare, IoT...).

OMEX is the natural ingestion engine for this pattern: it converts course text into structured grammar graphs with knowledge references, at scale, cheaply — and the evidence layer computes its metrics over those graphs deterministically, so the model interprets **structured evidence**, never raw prose, when classification decisions are made. (This pattern is host-side; OMEX's role is producing the graphs the evidence is computed from.)
