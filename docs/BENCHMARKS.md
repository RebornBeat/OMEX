# The Benchmark Program

Purpose: **replace every derivation in docs/PERFORMANCE.md with a measurement**, and settle every design question the docs currently mark open. Nothing in OMEX's performance story is considered established until it appears here as a result.

## 1. Baselines

- **The LLM pipeline itself** (the graph factory driven step-by-step with validation) — the primary end-to-end baseline for speed, loops-to-completion, energy, and quality per validated structure.
- **Established neural parsers** on standard treebanks — structure-accuracy baseline.
- **Dedicated grammar-correction systems** — correction baseline.
- **Entity-linking systems** — linker baseline (adapted to fabric paths).
- **Data-to-text / meaning-representation-to-text systems** and the **Tier-1 rendering stand-in** — realizer baselines (Tier 1 is the *fair* fluency baseline: same content, different renderer).

## 2. Metrics

**Parser:** labeled/unlabeled attachment accuracy over the full taxonomy; span exactness; property accuracy (tense/number/…); boundary F1 (sentence/paragraph/section); knowledge-span detection F1. **Correction head:** F0.5 on edit sets; meaning-preservation checks. **Linker:** precision/recall@k; margin calibration; re-resolution rate under the pool-connectivity check. **Realizer:** parse-back coverage (→100%); **addition rate (→0, hard gate)**; property-agreement rate; human fluency and voice-fidelity ratings vs. Tier 1. **System:** end-to-end latency per checkpoint; energy per validated structure and per response (J); resident memory; loops-to-completion vs. the LLM pipeline; throughput under batching.

## 3. The Open Design Questions as Experiments

1. **Joint vs. separate heads** (structure+correction+knowledge-span in one encoder vs. split models) — accuracy, latency, memory, at matched total parameters.
2. **Per-language vs. multilingual** — matched-parameter comparison per language tier (high/mid/low-resource), plus distillation-path evaluation.
3. **Realizer Paths A/B/C** — fluency/fidelity/latency triangle; C expected to win, measured anyway.
4. **Quantization ladder** — int8 vs. int4 per head; classification heads expected int4-tolerant, generation less so.
5. **Frame vs. tree granularity defaults** — fluency and fidelity per response type.
6. **OMEX-native vs. LLM-pipeline extraction** — the headline: compute, energy, wall-clock, and quality per validated structure, on identical inputs.

## 4. Device Matrix

Phone-class ARM (CPU-only) · single-board computer (e.g., Pi-class) · laptop CPU · consumer GPU · server GPU. Every system metric reported per device tier; batching curves (latency vs. batch size) per tier.

## 5. Protocol and Publication

Fixed evaluation corpora (held out from all training, provenance-logged); seeds and configs published; energy measured at the wall where possible, else by vendor counters with methodology stated; every result that supersedes a derivation is cross-referenced into docs/PERFORMANCE.md, which shrinks as this document grows. **Success criteria (v0 gates):** parser within striking distance of established parsers on calibration treebanks; addition rate ≈ 0 with coverage ≈ 100% for the Realizer path chosen; end-to-end structure extraction ≥ 3 orders of magnitude cheaper than the LLM pipeline per validated structure; full family resident and interactive on the single-board tier.
