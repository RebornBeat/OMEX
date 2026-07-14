# Deploying OMEX

## 1. Deployment Profiles

- **Embedded/standalone:** an application links the models directly (parser alone is already useful: structure, correction, knowledge-span detection). Minimum viable footprint: parser + Tier-0 renderer.
- **Orchestrated host:** the full loop — parser → intent traversal → knowledge pools → response graphs → rendering ladder — inside an orchestrator with a knowledge fabric (reference: Ozone-Studio + ZSEI; generic requirements in docs/INTEGRATIONS.md §4).

## 2. Memory Tiers (Generalized)

| Tier | Contents | Why |
|---|---|---|
| Always resident | All OMEX weights; fabric structural index (mmap/OS page cache); linker ANN index; hot caches | Weights are small and fixed; structural hops must be O(1); ANN is the latency-critical path |
| On demand | Per-entry fabric records; knowledge-pool neighborhoods (prefetched the instant a knowledge span fires) | Only hot neighborhoods matter; a residency cap bounds RAM |
| Write-behind | Emitted graphs; inferred edges; flywheel rendering pairs | Parsing must never block on storage; batch per window |

RAM scales with the **working set, not the corpus** — the fabric can grow indefinitely while device memory stays flat.

## 3. Loading, Versioning, Updating

- **Contract checks at load:** producer/consumer schema-version compatibility verified before any pipeline is assembled; mismatches fail fast with the exact schema delta.
- **Independent model swaps:** graph contracts mean the parser, linker, and realizer upgrade independently — swap one, re-verify contracts, done.
- **Adapter selection:** structural adapters chosen at load by task profile (code formatting, notation, citation styles) — syntax only, per the Golden Rule.
- **Fabric ops:** linker ANN index refresh on fabric-growth cadence; linker fine-tune triggered by margin-degradation thresholds; resolution caches TTL-invalidated on entry updates.

## 4. Monitoring (Production Signals)

Confidence distributions per head; **parse-back audits on a sampled fraction of all rendered responses** (coverage%, addition-rate — the hallucination sentinel, expected ≈0); linker margins and re-resolution rates; unresolved-span rates per domain (fabric growth queue); per-stage latency traces (parser / linker / traversal / assembly / render); escalation rates up the rendering ladder; energy counters per device tier.

## 5. The Fallback Ladder (Uniform Pattern)

```
OMEX model (Tier 2 where applicable)
   │ low confidence / audit failure
   ▼
Verification / re-decode with constraints
   │ still failing
   ▼
LLM stand-in for that single step (rendering, ordering, novel shapes)
   │ unavailable
   ▼
Deterministic floor (template renderer; rule-based ordering)
```

Every escalation is logged as training signal — the flywheel by which the OMEX tiers absorb the escalations over time.

## 6. Privacy Note on the Flywheel

Operational flywheel pairs (graphs → text) may contain user content. Deployments must gate flywheel collection behind explicit policy (consent, on-device-only training, or synthetic-only corpora), and provenance metadata must mark corpus origin so any trained model's lineage is auditable.
