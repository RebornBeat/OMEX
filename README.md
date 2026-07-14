# OMEX: Omni-Execution Format

**OMEX is a model format, a training methodology, and an execution interface for graph-native language models** — small, fast, auditable models that read and write explicit linguistic structure (grammar graphs) instead of hiding grammar, knowledge, and reasoning inside billions of opaque parameters.

If you have never encountered this project before, the one-paragraph version is this: today's large language models (LLMs) are enormous because they learn *everything at once* — grammar, style, world knowledge, reasoning — entangled across their weights. OMEX un-entangles them. It uses existing large models **once, as factories**, to produce hundreds of thousands of complete, validated *grammar graphs* — explicit structures describing exactly how every sentence is built, how sentences relate, what was grammatically wrong and how it was corrected, and which spans refer to knowledge topics. Small models are then trained directly on those graphs. The result is a family of models, each a fraction of a gigabyte, that natively **parse** text into structure, **correct** grammar, **reference** external knowledge without containing it, and **write** fluent text back out by running the same mapping in reverse — because writing is grammar traversal in reverse.

New to all of this? Start with **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — it explains, from zero, what traditional NLP was, what SLMs/LLMs actually learn, what a grammar graph is, and why separating grammar from knowledge changes the economics of language AI.

## The Three Eras

| Era | Grammar | Knowledge | Cost |
|---|---|---|---|
| **Traditional NLP** (pre-2018) | Explicit — hand-coded rules and statistical stages | External or absent | Brittle, narrow, plateaued |
| **SLM / LLM** (2018–present) | Implicit — emergent, distributed across parameters | Embedded in the same parameters | Enormous size, energy, opacity |
| **OMEX** | Explicit **and learned** — the model's native output *is* the graph | External, referenced by address — never embedded | Small models, auditable structure, local-first |

Traditional NLP decomposed language into components, but every component was hand-crafted or narrowly statistical, so the field plateaued. LLMs dissolved the components into one implicit soup — powerful, but at the cost of carrying all of civilization's knowledge inside the weights just to conjugate a verb. **OMEX closes the loop:** NLP's decomposition, reborn, with each component a trained neural model carrying LLM-grade linguistic understanding — possible *only because* LLMs exist to manufacture the training graphs.

## The Golden Rule

> **Weights are closed. Knowledge is data.**
>
> OMEX models never contain world knowledge, and an OMEX system never dynamically loads "knowledge-specific" model parameters. The model family (Parser, Linker, Realizer) is structurally complete and static. When a task needs domain knowledge, that knowledge is fetched as **external graph data** from a knowledge fabric, bounded by grammar-anchored traversal, and injected as content into a **Response Graph**. The only dynamic parameters permitted are **structural adapters** (syntax formatting — code blocks, citations, notation), never facts.

The full treatment — including how knowledge selection stays *bounded* rather than dumping entire topic subtrees — is in **[docs/KNOWLEDGE.md](docs/KNOWLEDGE.md)**.

## What an OMEX System Does

```
Input Text
    │
    ▼
Grammar Graph          ← OMEX GrammarParser (structure + correction + knowledge spans)
    │
    ▼
Intent / Meaning       ← host traverses the graphs (deterministic, inspectable)
    │
    ▼
Knowledge Context Pool ← bounded traversal of the external fabric, anchored by grammar
    │
    ▼
Decision & Task Work   ← host executes; results are verified content
    │
    ▼
Response Graph         ← assembled: structure + filled content + provenance + voice
    │
    ▼
Grammar Graph          ← same schema as the input side
    │
    ▼
Output Text            ← rendered (OMEX Realizer, or the rendering ladder during bootstrap)
```

Grammar sits on **both sides**. Meaning lives in the middle, as graphs — inspectable, cacheable, carrying provenance for every claim. Internal reasoning never generates text; text exists only at user-facing checkpoints.

## The OMEX Model Family

| Model class | Input | Output | Contains |
|---|---|---|---|
| **GrammarParser** | Text | Grammar graph | Grammar skill only — structure, correction, knowledge-span detection, boundaries |
| **KnowledgeLinker** | Span + context | Ranked fabric addresses | Addressing skill only — it knows *where* things live, never *what* they are |
| **Realizer** | Response graph (+ voice) | Text | Rendering skill only — it renders what the graph contains and nothing else |
| **Modality structural models** | Code, math, etc. | Modality graphs | The same thesis per formal structure (see docs/MODALITIES.md) |

Three skills. Zero facts. Indicative family size: under ~0.75 GB at int8, fully resident simultaneously on a phone or single-board computer.

## Document Map

| Doc | Read it for |
|---|---|
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Background from zero: NLP → SLM/LLM → OMEX, grammar graphs, glossary |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Model classes, heads, contracts, composition rules |
| [docs/FORMAT.md](docs/FORMAT.md) | The `model.omex/` package, grammar schema, interchange JSON, worked examples |
| [docs/KNOWLEDGE.md](docs/KNOWLEDGE.md) | **The Golden Rule; the five-step knowledge lifecycle; bounded knowledge traversal** |
| [docs/RESPONSE_GRAPH.md](docs/RESPONSE_GRAPH.md) | **The Response Graph: schema, assembly, invariants — the reasoning→rendering handoff** |
| [docs/REALIZER.md](docs/REALIZER.md) | **The highest-risk component: architecture, fidelity mechanisms, the rendering ladder** |
| [docs/MODALITIES.md](docs/MODALITIES.md) | Code, math, and perceptual modalities: what's native, what's bounded |
| [docs/LANGUAGES.md](docs/LANGUAGES.md) | Universal schema, per-language models, the multilingual bootstrap path |
| [docs/TRAINING.md](docs/TRAINING.md) | Corpus sources, the master generation prompt, validation rules, curriculum |
| [docs/EXECUTION.md](docs/EXECUTION.md) | The runtime loop, AMT-style reasoning, CPU/GPU optimization |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Speed/memory/energy comparisons — derivations with the arithmetic shown |
| [docs/RISKS.md](docs/RISKS.md) | The full risk register with mitigations and detection signals |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | The formal measurement program that replaces every derivation |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Profiles, memory tiers, versioning, monitoring, the fallback ladder |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Host systems: Ozone-Studio and ZSEI primers; what any host must provide |
| [docs/FAQ.md](docs/FAQ.md) | Every question, plainly answered — including all open design questions |

## What OMEX Is Not

- No embedded "optimization intelligence," no execution-optimizer binaries, no optimization fingerprints.
- No hardware profiles or per-device strategies in the format — hardware belongs to the runtime.
- No adaptive chunking — input windowing is the host's concern.
- No per-node memory budgets in execution plans.
- **No knowledge in the weights, ever — and no knowledge-specific parameters loaded at runtime.** See the Golden Rule.

## Status and Roadmap

- [x] Conceptual specification, schema alignment, knowledge-path formalization (this document set)
- [ ] Training corpus generation at scale (validated graph pipeline as data QA)
- [ ] GrammarParser v0 — single language, shared encoder, structure + correction heads
- [ ] KnowledgeLinker v0 — span → fabric-path resolution with refreshable index
- [ ] Tier-0 template renderer + Tier-1 rendering stand-in (day-zero response capability)
- [ ] Realizer v0 — trained from the operational flywheel + synthetic corpus
- [ ] Benchmark program (see docs/BENCHMARKS.md) — replaces all derivations with measurements
- [ ] Per-language line-up; Code-OMEX and Math-OMEX families

## License

MIT License. See LICENSE for details.
