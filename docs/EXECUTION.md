# OMEX Execution

## 1. The Runtime Loop

OMEX is **prompt-first** and **local-first**: execution begins from a prompt; all inference happens on the local device; the model family is small enough to stay fully resident.

```
1. Prompt/Request arrives (from a user or a host/orchestrator).

2. GrammarParser → grammar graph (structure + corrections + knowledge spans
   + boundaries), one forward pass per window; sentences batched in parallel.

3. KnowledgeLinker → each knowledge span resolved to ranked addresses in the
   external knowledge fabric (batched ANN search).

4. Intent traversal → meaning built by walking the graphs deterministically
   (§2 below — Stage 1 / Stage 2 / promotion; see also docs/INTEGRATIONS.md
   for the reference host's full AMT).

5. Bounded knowledge traversal → for each intent branch, a Knowledge Context
   Pool is built from the resolved entries, anchored by the query's own
   grammar and stopped at the relevance frontier (docs/KNOWLEDGE.md §3).
   Knowledge is fetched by reference and bounded by traversal — never
   loaded wholesale.

6. Decision & task execution → whatever work the intent requires; results
   are verified content.

7. Response Graph assembly → the host composes the structural skeleton,
   pre-filled content nodes (pool facts + task results, each with
   provenance), discourse relations, and voice properties
   (docs/RESPONSE_GRAPH.md).

8. Rendering → the rendering ladder turns the Response Graph into fluent
   text (§3 below; docs/REALIZER.md). Grammar on the way out, exactly as
   on the way in.
```

Task-checkpoint responses fall out naturally: whenever an intent branch reaches a completion or clarification checkpoint, steps 7–8 run for that branch's neighborhood — and because knowledge references ride on the sentence graphs from steps 2–3 and pools were built in step 5, the branch's applicable knowledge is already aggregated on it.

## 2. Reasoning Without Text

The system's "thinking" is **deterministic graph traversal — no model reasons in natural language, and no thought-tokens are ever generated.** This is a structural difference from LLM chain-of-thought, and it is both a correctness and an efficiency win: every inference step is inspectable, and internal reasoning costs zero decode steps.

The internal process, in five stages:

1. **Context pool accumulation (Stage 1).** Start from sentence nodes; walk typed grammar edges (Subject→Predicate, Verb→Object, prepositional/temporal/conditional attachments), coreference chains, and cross-sentence relationship edges; accumulate a growing pool of nodes, properties, and knowledge references that belong together. Expansion follows the traversal invariants (docs/CONCEPTS.md §5) — evidence, not similarity.
2. **Neighborhood continuity evaluation (Stage 2).** Decide whether adjacent pools merge or stay separate using *evidence of continuity*: shared entities via coreference, dependency links, discourse relations, ownership/temporal boundaries. This is what keeps two projects that both say "build" from collapsing into one intent.
3. **Promotion.** Sufficiently coherent neighborhoods become **intent nodes** — each carrying its canonical summary, actor/action/target/constraints, its full supporting evidence, and (via step 5 of §1) its Knowledge Context Pool. The intents form a tree: root goal → branches → leaves.
4. **Task creation and planning.** Concrete tasks derive from intent branches, each linked back to its branch — dependencies, loops, and sub-steps modeled explicitly.
5. **Response Graph assembly at checkpoints.** When a branch needs a user-facing response, its pool + results assemble into a Response Graph deterministically (docs/RESPONSE_GRAPH.md §4). Only here does text come into existence — rendered, in step 8.

A worked micro-example: *"Compare annealing and tempering for steel."* Stage 1 builds two pools (one per process); Stage 2 keeps them distinct but related (parallel processes, shared target entity "steel"); promotion yields two sibling intents under a comparison root; knowledge traversal fills each with effect-edges connecting to steel; the comparison itself is **edge comparison over the two pools** — no sentence is generated to "think" it; the Response Graph then encodes the comparison structure (parallel frames + a Contrast relation), and rendering produces the prose. Every claim in the output traces to a specific fabric edge.

## 3. Operating Before the Realizer Exists — the Rendering Ladder

The loop above never depends on the Realizer's existence; it depends on the **Response Graph contract**, and three renderers satisfy it:

- **Tier 0 — template renderer.** Deterministic: walks the skeleton, emits content in grammatical order via per-language rule tables (constituent order, agreement from node properties, default connectives per discourse relation). Correct, plain, robotic. Zero ML. Exists from day zero; can never be unavailable.
- **Tier 1 — LLM rendering stand-in.** The graph-factory model renders the Response Graph under a render-only instruction (function words, inflection, and declared-relation connectives permitted; content changes forbidden). Every output is **parse-back validated** — full content coverage, zero ungrounded spans — or it is re-rendered / dropped to Tier 0.
- **Tier 2 — the OMEX Realizer**, once trained; escalates to Tier 1 on low confidence or parse-back failure; Tier 0 is the unbreakable floor.

**Content is fixed at assembly time and identical across tiers; only fluency climbs the ladder.** And every validated Tier-0/Tier-1 rendering becomes Realizer training data — the operational flywheel (docs/TRAINING.md §1C, docs/REALIZER.md §6).

## 4. Why This Is Fast: The Decisive Structural Facts

1. **Non-autoregressive structure prediction.** A parser with biaffine-style heads emits an entire sentence's tree in **one** forward pass. A general LLM emitting the same tree as JSON performs **hundreds of sequential decode steps**. This one design choice outweighs every kernel trick combined: it converts an O(output-tokens) sequential process into O(1) passes.
2. **Sentence-level parallelism.** Once boundaries are tagged, grammar extraction is embarrassingly parallel across sentences. Batch them.
3. **Everything stays resident.** The whole family at int8 is a fraction of a gigabyte: no offloading, no swapping, no layer streaming. Load once; the working set is stable.
4. **Reasoning generates zero tokens.** Thinking is traversal (§2). The only autoregressive work in the entire loop is the Realizer's short, final surface strings — honestly noted: the Realizer is the slowest model in the family per invocation, and it still decodes tens of tokens from a sub-300M model once per checkpoint, versus an LLM decoding its reasoning *and* its answer at billions of parameters per token (arithmetic in docs/PERFORMANCE.md §2).

## 5. CPU Execution

- Small-model single-stream inference is **memory-bandwidth bound**: throughput ≈ bandwidth ÷ bytes touched per pass. Optimize by shrinking bytes (int8 → int4 where classification accuracy holds — structure/correction heads tolerate quantization far better than open-ended generation) and by **batching** (weights are read once per batch, amortizing bandwidth across many sentences).
- Use SIMD-optimized integer kernels (AVX2/AVX-512/AMX on x86; NEON/SME on ARM).
- Thread strategy for tiny models: parallelize **across batch items**, not within layers — intra-layer parallelism of a ~100M model underutilizes cores and pays synchronization costs. Pin threads; on multi-socket machines keep weights and workers NUMA-local.
- Tokenization and span bookkeeping are cheap pure-CPU work: run them on separate threads, pipelined ahead of encoder batches.

## 6. GPU Execution

- For tiny models, **kernel-launch overhead can rival compute**. Record the whole forward pass as one launch (CUDA Graphs or equivalent); fuse elementwise ops; prefer a few large batched matmuls over many small ones.
- Batch aggressively — hundreds of sentences per pass is the natural operating point; per-sentence latency collapses.
- Keep tokenized batches device-resident across stages; **transfer graphs, not activations** — the graphs models exchange are tiny.
- Multi-model concurrency: parser, linker, and realizer are small enough to share one GPU via streams; or split — parser on GPU, linker ANN search on CPU, since ANN is pointer-chasing, not matmul.

## 7. Multi-Model Orchestration

- **Shared encoder, many heads:** one encoding, N task outputs — the highest-leverage form of "running together."
- **Pipeline overlap:** while sentence batch *k* is in the structure head, batch *k+1* tokenizes and batch *k−1*'s knowledge spans are in the linker; knowledge-pool traversal for earlier branches runs concurrently on CPU. CPU work (fabric traversal, span bookkeeping, ANN) and GPU work (encoder passes) run as a two-lane pipeline, joined at window boundaries.
- **Cascade/escalation:** run the OMEX model first; escalate to a general SLM/LLM only on low-confidence outputs — the same pattern the rendering ladder embodies. This keeps quality guarantees while paying large-model cost only on the hard tail; every escalation is logged as training signal.

## 8. What Dominates, Honestly

In any deployment that still uses an LLM-driven pipeline for graph generation, **that pipeline dominates all cost** (Amdahl's law). The optimization program that matters most is therefore: generate corpora → train the parser → swap it in. Every kernel-level optimization above is second-order next to that swap.
