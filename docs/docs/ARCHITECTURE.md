# OMEX Architecture

## 1. Design Rules

1. **The graph is the interchange contract.** OMEX models compose by exchanging *typed graphs*, never hidden neural states. Every intermediate is inspectable, cacheable, and reusable as training data.
2. **Grammar in the weights; knowledge by reference.** No OMEX model contains world knowledge. Knowledge-bearing spans are detected in the model and resolved outside it.
3. **Small and resident.** Every model is sized to live permanently in memory on ordinary devices. No offloading, no swapping, no layer streaming.
4. **Declared contracts.** Every model declares exactly one input/output contract — `text → graph`, `graph → text`, `graph → graph`, or `span+context → references` — pinned to schema versions. Two models compose if and only if the producer's output contract matches the consumer's input contract at compatible versions. **The Response Graph is a first-class interchange structure under these contracts**: `response-graph → text` is the Realizer's declared contract, and any renderer on the rendering ladder must satisfy it identically (docs/RESPONSE_GRAPH.md).

## 2. The Boundary: Weights vs. Knowledge Data (Normative)

> OMEX models never contain world knowledge, and an OMEX system never dynamically loads "knowledge-specific" model parameters. The OMEX model family (Parser, Linker, Realizer) is structurally complete and static. When a system requires domain-specific knowledge, that knowledge is fetched as **external graph data** from the host's knowledge fabric, bounded by grammar-anchored traversal (docs/KNOWLEDGE.md), and injected as content nodes into the Response Graph. The OMEX Realizer renders this graph using its universal grammar weights. The only dynamic parameters OMEX permits are **structural adapters** (LoRA-class), which adjust the model for specialized *syntax formatting* — code blocks, legal citations, notation conventions — never for domain facts.

This section is normative: a system that loads knowledge parameters, however partially, is out of specification. The reasoning — hallucination vector, currency, RAM ceiling, per-domain management — is given in docs/KNOWLEDGE.md §1 and docs/CONCEPTS.md §7. Note that even the KnowledgeLinker contains no knowledge: it contains an *addressing skill* only.

## 3. GrammarParser

The workhorse: **text → grammar graph**, in a single forward pass.

**Encoder.** A compact transformer encoder (indicative range: 100–300M parameters; final sizes are benchmark outcomes).

**Heads** — all sharing the encoder's token representations, so one encoding pass serves every task:

- **Boundary head.** Token-level tagging of sentence boundaries, paragraph open/close signals, and section markers with level. One pass per input window.
- **Structure head.** Per-sentence grammar-tree prediction, **non-autoregressive**: biaffine-style arc-and-label scoring over the full grammatical taxonomy, decoded under hard constraints — every child's span contained in its parent's, single parent per node, sibling spans non-overlapping, every surface token on exactly one leaf path. Node properties (tense, aspect, mood, voice, person, number, gender, case, definiteness, comparison, polarity, subtype) are per-node classifications.
- **Correction head.** Tagging-based grammatical error correction: per-token edit operations (keep / replace / insert / delete / transform) that reconstruct the corrected sentence from the original. Tagging-style correction is proven to work well in small models and decodes in the same single pass.
- **Knowledge-span head.** Tags knowledge-bearing spans with a coarse kind (`concept | principle | procedure | entity | process | tool | material`). **Detection only** — resolution belongs to the KnowledgeLinker.

**Why heads, not separate models.** Detection and correction share nearly all linguistic representation — you must parse to correct. One encoder pass amortizes across all tasks; heads can be switched off per call (structure-only when correction isn't needed). *(Joint vs. separate remains an explicit benchmark item — see docs/BENCHMARKS.md — with joint as the strongly expected winner.)*

## 4. KnowledgeLinker

A separate, smaller model: **span-in-context → ranked knowledge-graph paths.**

Design: a bi-encoder. One side embeds the detected span with its sentence context — including its governing verb and co-arguments, which is what resolves ambiguity ("eat" pulls "Apple" toward the fruit; "invest" toward the company) — reusing the parser encoder's representations, so no second embedding model runs. The other side is a precomputed embedding index (HNSW) over the external knowledge fabric's entries. Resolution = approximate nearest-neighbor search → candidate paths → optional zero-shot verification for low-margin cases. A second, structural safety net exists downstream: a wrong resolution yields a Knowledge Context Pool with near-zero co-anchor connectivity, which triggers re-resolution with the next-ranked candidate (docs/KNOWLEDGE.md §3).

**Why separate.** The knowledge fabric grows continuously. When it grows, you refresh the index (cheap) or fine-tune the linker (moderate) — the parser, the expensive model, is never touched. This mirrors the proven decomposition in entity-linking research: *mention detection → candidate generation → disambiguation* — here applied to arbitrary knowledge-graph paths rather than a fixed encyclopedia.

## 5. Realizer

**Response graph → text** — the highest-risk model in the family, with its own dedicated document: **docs/REALIZER.md**. The architectural summary:

**The fundamental asymmetry.** The Parser is non-autoregressive — a sentence's structure is a static map, all arcs predictable simultaneously. The Realizer **must be autoregressive** — fluent text is sequential decision-making: the word chosen at step *N* constrains the grammatical options at step *N+1*. This asymmetry is accepted, not fought — and it is cheap in context, because the Realizer generates only short, final, user-facing surface strings, never thought-tokens and never structure-as-JSON.

**The chosen architecture (Path C).** Graph encoder (graph transformer/GNN over the Response Graph — node types, content, properties, discourse roles; structural relations as attention biases) → autoregressive transformer decoder with a **voice prefix** (tone, formality, warmth, directness, humor embedded and prepended) → **coverage tracking** (every content node covered exactly once — the classical remedy for omission and repetition) → **constrained decoding** (inflection and agreement forced to match node properties; tree-granularity sentences additionally constrain phrase order) → **cycle-consistency training** (the frozen parser re-parses every training output; divergence from the input graph's content set is penalized, with **additions** penalized hardest — the parser is the Realizer's permanent adversarial auditor).

**Training asymmetry.** The parser trains on original *and* corrected sentences; the Realizer trains **only on the corrected side** (graph → corrected sentence) — it must never learn to emit errors. Every parser training pair, reversed, is a Realizer training pair; voice-variant pairs are constructed content-leaf-identical so voice can never learn to alter facts.

**Operational independence.** The AMT and orchestrator depend on the Response Graph *contract*, not on the Realizer's existence: a deterministic template renderer (Tier 0) and a parse-back-validated LLM stand-in (Tier 1) satisfy the same contract from day zero, and every validated rendering they produce becomes Realizer training data — the rendering ladder and its flywheel, fully specified in docs/REALIZER.md §6.

## 6. Language Strategy

**Universal schema, per-language models** (recommended default). The node taxonomy and property set are language-general — one schema, per-language realization, in the spirit of universal grammar annotation projects. Per-language parsers at equal size are smaller, faster, and generally more accurate than multilingual ones, except for low-resource transfer. A tiny language-ID router selects the model; every downstream consumer sees one schema regardless of language.

Multilingual variants remain sanctioned as a **bootstrap** for low-resource languages, distilled later into per-language models. *Per-language vs. all-languages-together is one of the open design questions to be settled empirically — full treatment in docs/LANGUAGES.md; the benchmark program decides.*

## 7. Modality Strategy

The same approach generalizes wherever structure is formal — full treatment in docs/MODALITIES.md:

- **Code and math — full applicability, likely better than text.** Their grammars are strictly closed. A code-structural model emits the AST plus a semantic edge layer (Calls, DataFlows, ImplementsPattern, DependsOn, ...); a math model emits derivation structure (LogicallyImplies, Generalizes, UsedToProve, ...). This is the OMEX thesis in its purest form.
- **Image, audio, video — the relational layer only, with an honest boundary.** Spatial relationships, temporal chains, object interactions, prosody events — these are learnable graph targets once perception has produced entities. But *perception itself* (pixels → objects, waveform → events) is not a closed rule system and requires perceptual models outside OMEX's thesis. Design consequence: perceptual pipelines pair a perception front-end with an OMEX relational model; code/math need no front-end at all.

## 8. Composition Semantics

Three sanctioned modes — and one explicitly out of scope:

1. **Sequential graph pipelines** (default). Model A's output graph is model B's input. Matches any orchestrated pipeline architecture.
2. **Shared-encoder multi-head.** One encoder pass, many task outputs simultaneously. The genuine form of "running together."
3. **Adapter composition.** Domain/register adapters (LoRA-class low-rank add-ons) selected at load time on a common backbone — the one well-behaved way to "expand" a fixed model, and per §2, permitted for *syntax* only.
4. ***Out of scope:* runtime weight expansion or merging across heterogeneous trained models.** Trained transformers have fixed weight matrices; the known growth/merge techniques (MoE expert addition, Net2Net growth, model soups/TIES merging) either require training or are same-architecture-only and brittle. OMEX's answer to interoperability is the **graph contract**: when a task needs text understanding *and* code understanding, the text parser emits a grammar graph, the code model emits a code graph, and the host stitches them with cross-modal edges (e.g., `DescribedBy`, `ImplementedIn`). Execution contexts expand through graph stitching; parameters never physically merge.

---
