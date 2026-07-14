# OMEX Performance: Speed, Memory, Energy

## 0. Methodology — Read This First

**Nothing in this document is a measured benchmark.** OMEX v0 models have not shipped; the formal measurement program is specified in **docs/BENCHMARKS.md, which supersedes this document result-by-result as measurements land** — every derivation below is a placeholder awaiting its measurement. What this document provides is **derivations**: arithmetic consequences of parameter counts, decoding modes, and memory arithmetic, with every step shown so you can check it. Earlier drafts of this project circulated tables with figures like "96% faster" and "99% energy saved" presented as results; those were projections dressed as measurements and are **withdrawn as such** — the honest versions of those same claims, derived and labeled, are below.

## 1. Compute per Extracted Structure

A dense transformer's forward pass costs ≈ 2 × (parameter count) FLOPs **per token processed or generated**.

**LLM route.** A 7B general model asked to emit a ~600-token JSON grammar analysis of one sentence, autoregressively:

```
2 × 7×10⁹ × 600 ≈ 8.4 × 10¹² FLOPs  (8.4 TFLOPs)   + prompt-processing cost
```

**OMEX route.** A 100M-parameter GrammarParser, non-autoregressive, on a 30-token sentence:

```
2 × 1×10⁸ × 30 ≈ 6 × 10⁹ FLOPs  (6 GFLOPs)
```

**Ratio: ≈ 1,400× less compute for the same sentence's structure** — before counting the LLM route's retries and repeated validation loops (the reference generation pipeline validates every structure with multiple independent confirmations), which multiply the gap toward **4–5 orders of magnitude per validated structure**.

## 2. The Realizer's Cost — the Honest Line Item

The Realizer is the one autoregressive model in the family, so it deserves its own derivation rather than being averaged away.

**OMEX response rendering.** A ~200M-parameter Realizer rendering a typical checkpoint response of ~3 sentences ≈ 60 tokens:

```
2 × 2×10⁸ × 60 ≈ 2.4 × 10¹⁰ FLOPs  (24 GFLOPs per response)
```

**LLM route for the same response.** A 7B model generating just those 60 answer tokens:

```
2 × 7×10⁹ × 60 ≈ 8.4 × 10¹¹ FLOPs  (840 GFLOPs)  →  ~35× more, on the surface string alone
```

— and the honest comparison is worse for the LLM, because it also generates its *reasoning* (chain-of-thought commonly runs hundreds to thousands of tokens per answer), while OMEX reasoning is traversal and generates **zero** tokens (docs/EXECUTION.md §2). Counting reasoning tokens, the end-to-end response-side gap is **2–3 orders of magnitude**. Within the OMEX family itself, the Realizer remains the slowest model per invocation — an accepted asymmetry (docs/REALIZER.md §2), invoked once per checkpoint on short strings.

## 3. Energy

On the same hardware, energy scales approximately with FLOPs, so the ratios above carry directly to joules. As an order-of-magnitude illustration (modern accelerators deliver very roughly 10⁻¹²–10⁻¹³ joules per FLOP effective):

```
LLM route:   8.4 TFLOPs  →  roughly 1–8 J per analyzed sentence
OMEX route:  6 GFLOPs    →  roughly 0.001–0.006 J per analyzed sentence
```

Three-plus orders of magnitude less energy per structure — the same conclusion by a second path. Where "per token" is the unit of comparison: an LLM pays ~2N FLOPs for every output token, hundreds of times per sentence analysis; the OMEX parser pays for the input tokens once. On the response side, the §2 arithmetic gives the corresponding joule ratios for rendering.

## 4. Memory

```
7B model, int8:                    ≈ 7.0 GB weights
Full OMEX family, int8:
  GrammarParser  100–300M          ≈ 0.10–0.30 GB
  KnowledgeLinker 50–150M          ≈ 0.05–0.15 GB
  Realizer       100–300M          ≈ 0.10–0.30 GB
  ─────────────────────────────────────────────
  Total                            ≈ 0.25–0.75 GB
```

**~90–96% reduction versus a single 7B SLM; ~98–99% versus a 70B-class model** — and, more importantly, the entire family is *simultaneously resident* on a phone or a Raspberry Pi with no swapping, offloading, or layer streaming. Because knowledge lives outside the weights, **RAM scales with the working set, not the corpus**: the external knowledge fabric can grow to billions of entries while device memory stays flat — and per-query knowledge use is bounded by the Knowledge Context Pool mechanism (docs/KNOWLEDGE.md §3), not by fabric size.

## 5. Latency and Throughput

- **Single-stream CPU bound (bandwidth math).** Small-model inference is memory-bandwidth bound: passes/s ≈ bandwidth ÷ bytes touched per pass. A 100M int8 model on a 50 GB/s DDR5 system: 50×10⁹ ÷ 1×10⁸ ≈ **~500 forward passes/s upper bound, single-stream** — hundreds of sentences per second, CPU-only, before batching.
- **Startup.** Loading <1 GB of weights vs. tens of GB: cold-start drops from tens of seconds (large models with offloading) to well under a second. No warm-up analysis phase exists — the format contains no embedded optimizers to initialize (removed by design).
- **Structure latency.** One forward pass per sentence vs. hundreds of sequential decode steps: per-sentence structure latency drops from seconds (LLM JSON emission) to milliseconds (single pass), and collapses further under batching.
- **Response latency.** Realizer decode of tens of tokens from a sub-300M model per checkpoint; internal reasoning contributes traversal time only (native-code adjacency-list walks — see docs/RISKS.md R5 for the performance discipline), never decode time.

## 6. Quality-Side Comparisons (qualitative, structural)

| Dimension | Traditional NLP | SLM/LLM | OMEX |
|---|---|---|---|
| Grammar representation | Explicit, hand-coded, brittle | Implicit, emergent, entangled | **Explicit and learned** — native output is the graph |
| Adaptability to messy language | Poor | Excellent | Excellent (trained on LLM-manufactured graphs, including error/correction pairs) |
| Auditability | High | Near zero | **High** — every node carries spans; every claim carries provenance |
| Grammar correction | Separate rule systems | Implicit, unlocalized | **Intrinsic and localized** — per-span, typed edits |
| Knowledge | External or absent | Embedded, provenance-free (hallucination risk) | **External by reference** — addresses + bounded pools, never contents-in-weights |
| Determinism | Deterministic | Stochastic | Deterministic at temperature 0; schema-constrained by construction |
| Output structural validity | By construction | Not guaranteed (malformed JSON, drift) | **Guaranteed by constrained decoding** |
| Generation fidelity | n/a | Can invent content | Renderer emits **only** what the Response Graph contains — parse-back-auditable |

## 7. Format Comparison

| Feature | ONNX | GGUF | TorchScript | OMEX |
|---|---|---|---|---|
| Design philosophy | Interchange | Quantized LLM packaging | Traced export | **Graph-native language models** |
| Unit of exchange between models | Tensors | Tensors | Tensors | **Typed graphs** |
| Grammar | Implicit in weights | Implicit in weights | Implicit in weights | **Explicit, schema-versioned** |
| Knowledge | In weights | In weights | In weights | **External, referenced, pool-bounded** |
| Correction provenance | None | None | None | Original/corrected pairs, per span |
| Structure decoding | Autoregressive | Autoregressive | Autoregressive | **Single-pass, non-autoregressive** |
| Multi-model composition | Ad hoc | Ad hoc | Ad hoc | Graph contracts + shared encoders + adapters |

## 8. What Has Not Been Measured Yet

Parsing accuracy vs. established parsers on standard treebanks; correction quality vs. dedicated GEC systems; linker precision/recall at scale; realizer fluency, coverage, and **addition rate**; per-language vs. multilingual tradeoffs; joint vs. separate head tradeoffs; end-to-end energy on target devices. All are line items in **docs/BENCHMARKS.md**, whose results supersede this document section by section. This document shrinks as that one grows.
