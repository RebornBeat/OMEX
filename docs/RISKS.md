# Risk Register and Mitigations

Every material risk, its likelihood/impact, its mitigation, and — critically — its **detection signal**, because a risk you can detect early is half-solved.

## 1. Register (Summary)

| # | Risk | Likelihood | Impact | Primary mitigation |
|---|---|---|---|---|
| R1 | Realizer stiffness / infidelity | High | Medium/High | Coverage + cycle consistency + rendering ladder (docs/REALIZER.md) |
| R2 | Bootstrap corpus quality (GIGO) | Medium | High | Treebank calibration gate; validation loops; interleaved real data |
| R3 | Linker misresolution (the Apple problem) | Medium | High | Context-conditioned embedding; verification; pool-connectivity check |
| R4 | Knowledge fabric coverage gaps | Medium | Medium | Unresolved-span policy; graceful degradation; fabric growth loop |
| R5 | Orchestrator traversal performance | Medium | High | Native-code adjacency-list traversal; O(V+E); pool caching |
| R6 | Language proliferation cost | Medium | Medium | Demand-ordered rollout; shared generator; multilingual bootstrap |
| R7 | Schema evolution breakage | Low | High | Versioned schemas; contract checks at load; migration tooling |
| R8 | Host-dependency (adoption) | High initially | Medium | Minimal host requirements table; standalone embedding mode |
| R9 | Perception front-end dependency | High (perceptual only) | Medium | Honest boundary; front-end-agnostic entity contract |
| R10 | Derivations mistaken for benchmarks | Low (docs guard it) | High (credibility) | Loud methodology labels; benchmark program replaces derivations |

## 2. The Top Risks, in Detail

**R1 — Realizer.** Fully treated in docs/REALIZER.md. Note here only the systemic containment: because content is fixed at Response-Graph assembly, Realizer failure degrades *fluency*, never *facts* — the ladder guarantees a correct plain answer always exists. Detection: parse-back audits (coverage%, addition-rate) run continuously in production, not only in training.

**R2 — Bootstrap corpus quality.** If the LLM graph factory emits subtly wrong trees or missed relationships, the small models learn the errors. Mitigations: (a) the per-item validation discipline (repeated independent confirmation) as the first gate; (b) **calibration against gold-standard human-annotated treebanks — if factory agreement drops below ~95% on the calibration set, corpus generation pauses until prompts/models are fixed**; (c) interleaving real validated pipeline output with synthetic data; (d) versioned corpus provenance in `metadata.json` so any trained model's data lineage is auditable. Detection: rolling agreement metrics per batch; distribution drift monitors per node type.

**R3 — Linker misresolution.** "Hallucination by reference": right grammar, wrong address. Mitigations: the linker embeds the span **with its governing verb and co-arguments** (context decides fruit vs. company); zero-shot verification on low-margin candidates; and the **pool-connectivity check** — a wrong resolution yields near-zero co-anchor connectivity at traversal time, a loud signal that triggers re-resolution with the next candidate (docs/KNOWLEDGE.md §3). Detection: margin distributions; re-resolution rates; human audit of low-connectivity pools.

**R4 — Fabric coverage gaps.** A detected span may resolve to nothing (the fabric doesn't know the topic yet). Policy: the span is marked *unresolved*; the intent branch carries it as an explicit unknown; the Response Graph may state the gap ("no entry found for X") or ask a clarifying question — the system **degrades gracefully and honestly** rather than guessing. Every unresolved span is logged as a fabric-growth work item. Detection: unresolved-rate per domain.

**R5 — Orchestrator performance.** OMEX makes models fast; if graph traversal is slow, it becomes the bottleneck. Mitigations: traversal engines in native code over adjacency lists (never JSON-walking in the hot path); O(V+E) algorithms; aggressive caching of context pools and knowledge pools keyed to branch identity; the two-lane CPU/GPU overlap (docs/EXECUTION.md). Detection: per-stage latency accounting in every request trace.

**R6–R10** are contained as summarized above; R7 deserves one line more: **schemas are the compatibility currency**, so every load performs a contract check (producer output vs. consumer input at schema versions), and schema changes ship with migration notes — a graph corpus outlives any single model generation.

## 3. The Two Meta-Risks

**Drift back toward the LLM trap** — someone proposes "just put the common knowledge in the weights" or "let the Realizer fix awkward graphs." Both violate the Golden Rule and invariant 3 of the Response Graph. The docs now state these as invariants precisely so drift is detectable as a spec violation, not a matter of taste. **Credibility risk** — presenting projections as results. Contained by docs/PERFORMANCE.md's methodology section and the benchmark program; the rule stands: *derivations are labeled derivations until measurements replace them.*
