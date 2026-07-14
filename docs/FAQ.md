# OMEX FAQ

Plain-language answers to every question we've been asked — including the design questions that are deliberately still open. Where a question is open, the answer states the **recommended default**, the reasoning, and the fact that the benchmark program settles it.

### Q1. In one sentence, what is OMEX?

A format and methodology for small models that read and write **explicit grammar graphs** — parsing text into structure, correcting it, referencing external knowledge, and rendering structure back into text — instead of doing everything implicitly inside billions of parameters.

### Q2. How is this different from traditional NLP parsers (or tools like classic pipeline parsers)?

Traditional NLP components were hand-coded rules or narrow statistical models — a "dumb layer" that plateaued. OMEX components are **neural models trained on LLM-manufactured semantic grammar graphs**: they carry LLM-grade adaptability to messy, irregular, real language while keeping NLP's explicitness. NLP's decomposition, without NLP's brittleness — possible only because LLMs exist to manufacture the training graphs.

### Q3. Why not just use a small LLM?

Because a small LLM still entangles knowledge with grammar (so it's larger than the linguistic task requires, and hallucination-prone), still decodes autoregressively (so structure emission takes hundreds of sequential steps instead of one pass), and still gives you no auditable structure. OMEX gets you a smaller model, a single-pass parse, spans and provenance on everything, and knowledge by reference.

### Q4. Does an LLM actually "know" grammar? Isn't OMEX redundant?

LLMs do encode grammar — research shows syntax is recoverable from their internal representations. The difference OMEX makes is **explicitness, separability, and auditability**, not presence versus absence. In an LLM, grammar is entangled with everything and inspectable by no one; in OMEX, it is the native output format.

### Q5. Would this lead to a model with true semantic understanding of grammar — that actually understands language?

The **model** truly understands grammar: grammar is a closed system, closed systems are masterable by small models, and existing neural parsers already reach near-human agreement on structure at a fraction of LLM size — so feasibility is not speculative. "Understanding language" in full means structure **plus** knowledge and context — and OMEX deliberately splits that: the model owns structure; the **composed system** (parser + knowledge fabric traversal + intent building) is what understands language, with provenance at every step. Arguably a stronger position than the LLM's, because every step can be checked.

### Q6. Should grammar correction live inside the grammar model or be separate?

**Inside — as a head on the same model** (recommended default). You must parse to correct; detection and correction share nearly all representation; the training data pairs them intrinsically (every sentence carries original + corrected); a joint head decodes in the same single pass and can be switched off when not needed. A separate model would double inference for no benefit. *Open item: the joint-vs-separate variant is still benchmarked, per the project's verify-everything policy; joint is the strongly expected winner.*

### Q7. Should knowledge detection live inside the grammar model or be separate?

**Split by function.** *Detection* ("this span revolves around a knowledgeable topic, of roughly this kind") is cheap, local, stable — a head on the parser. *Linking* ("which exact knowledge-graph path") depends on a growing external fabric — a **separate** model (the KnowledgeLinker), so the fabric can grow without ever retraining the parser.

### Q8. Two-stage knowledge handling — general detection first, then a tying model?

Yes — exactly, and it matches the proven decomposition from entity-linking research: **mention detection → candidate generation → disambiguation**. Stage A (parser head): flag the span and coarse kind. Stage B (KnowledgeLinker): embed span-in-context, ANN-search the fabric's entry embeddings, optionally zero-shot-verify low-margin candidates. The payoff is maintenance: fabric grows → refresh the index or fine-tune the linker; the expensive parser is untouched. And because references ride on sentence nodes, intent branches downstream arrive with applicable knowledge **already aggregated**.

### Q9. So there ARE models around knowledge — do you load knowledge-specific models or parameters as needed?

**No — and this is the Golden Rule.** Weights are closed; knowledge is data. The family is structurally complete and static: no medical parser variant, no metallurgy adapter, no "load the physics head." Even the KnowledgeLinker contains no knowledge — only an *addressing skill* (it finds where "annealing" lives; it cannot state one fact about it). Your instinct that "loading only some knowledge still beats an LLM carrying everything" is right — and OMEX honors it **at the data tier**: per query, a bounded fragment of the knowledge graph is fetched and injected as content, which is strictly better than partial weight-loading because it's instant, current, provenance-carrying, and cannot be mis-remembered. Weight-tier loading — even partial — would reintroduce the hallucination vector, staleness, RAM creep, and per-domain model sprawl. The only dynamic parameters permitted are **structural adapters** (syntax formatting — code blocks, citations, notation), never facts. Full treatment: docs/KNOWLEDGE.md §1.

### Q10. A knowledge reference like `/HeatTreatment/Annealing` covers a huge topic — how is a response bounded rather than dumping the whole subtree?

Because **a reference is an entry point, not a payload.** The same Stage 1/Stage 2 methodology that builds text neighborhoods runs over the *knowledge graph*, anchored by the query's own grammar: "How does annealing **affect** steel?" yields the anchor `(Annealing, AFFECT, Steel)`, so traversal from the entry follows effect-class edges only, scores paths connecting *both* anchors highest, and stops at a **relevance frontier** when evidence weakens (no co-anchor connection, edge-class drift, depth-decayed score). The surviving subgraph — maybe six nodes out of hundreds — is the **Knowledge Context Pool** for that intent branch: bounded, scored, every fact carrying its address. That pool, not the raw address, is what "knowledge aggregated on the branch" means. Bonus: a *wrong* resolution (fruit-Apple in a finance question) produces a pool with near-zero co-anchor connectivity — a loud, detectable failure that triggers re-resolution. Full algorithm and worked example: docs/KNOWLEDGE.md §3.

### Q11. One model per language, or one multilingual model?

**Per-language models over a universal schema** (recommended default). The node taxonomy and properties are language-general — one schema regardless of language — while per-language parsers at equal size are smaller, faster, and generally more accurate than multilingual ones. A tiny language-ID router selects the model. Multilingual variants stay sanctioned as a **bootstrap for low-resource languages**, distilled later per-language. *This is an explicitly open design question — per-language vs. all-together is a benchmark line item; docs/LANGUAGES.md states the default and reasoning, the measurements decide.*

### Q12. Do modality-specific models (code, images, audio, video, math) get the same natural understanding as the text model?

**Code and math: yes — and likely better**, because their grammars are strictly formal and fully closed (a code model emits the AST plus semantic edges like Calls/DataFlows/ImplementsPattern; a math model emits derivation structure). **Image/audio/video: partially, with an honest boundary** — the *relational* layers (spatial relationships, temporal chains, object interactions, prosody events) are learnable graph targets, but *perception itself* (pixels → objects, waveform → events) is not a closed rule system and needs perceptual models outside OMEX's thesis. Perceptual pipelines pair a perception front-end with an OMEX relational model; code/math need no front-end at all. Full treatment: docs/MODALITIES.md.

### Q13. Fixed vs. expandable matrices — can OMEX models expand each other's parameters to run together?

The facts: standard trained transformers — all SLMs/LLMs — have **fixed** weight matrices post-training. Known expansion mechanisms (MoE expert addition, Net2Net-style growth, model soups/TIES merging) either require training or are same-architecture-only and brittle. **OMEX deliberately does not chase runtime weight surgery.** Its answer to "running together" is threefold and already in the format: **(1)** shared-encoder multi-head — genuinely one model doing many tasks in one pass; **(2)** adapter composition — a backbone legitimately "expanded" by low-rank structural adapters at load time; **(3)** sequential graph pipelines — models exchange **typed graphs**, not hidden states. When a task needs text *and* code understanding, each model emits its graph and the host stitches them with cross-modal edges: **execution contexts expand through graph stitching; parameters never physically merge.** The graph contract is the interoperability layer weight surgery was trying to be — and pipeline overlap recovers most of the throughput fused execution would give, without the fragility.

### Q14. How do you get a model to *respond* like an LLM after breaking the LLM into components?

By the principle the whole design rests on: **writing is grammar traversal in reverse**, and grammar sits on both ends — parser inbound, realizer outbound. The loop: Input Text → Grammar Graph → Intent Graph → Knowledge Pools → Decision/Execution → **Response Graph** → Grammar Graph → Output Text. The response graph is *assembled*, not imagined: content nodes from verified task results and knowledge pools, structure from the intent branch being answered, tone from voice properties on the nodes. The Realizer — trained on the parser's pairs reversed — renders it fluently. Task checkpoints trigger response-graph assembly for the relevant branch, whose evidence and knowledge are already aggregated on it. And it surpasses the LLM mechanism in one measurable way: **the renderer cannot hallucinate** — it renders only what the graph contains, and everything the graph contains has provenance.

### Q15. How does the system respond before the Realizer exists — or while it's still being trained?

Because the AMT depends on the **Response Graph contract, not on the Realizer** — and three renderers satisfy that contract: **Tier 0**, a deterministic template renderer (walks the skeleton, emits content in grammatical order via rule tables; correct, plain, robotic; zero ML; exists from day zero); **Tier 1**, the LLM rendering stand-in (the graph-factory model renders under a render-only instruction, every output parse-back validated for full coverage and zero additions); **Tier 2**, the trained Realizer, with escalation up the ladder on failure and Tier 0 as the unbreakable floor. **Content is fixed at assembly and identical across tiers; only fluency climbs.** And the flywheel: every validated Tier-0/Tier-1 rendering is a `(response graph → text)` training pair — **operating the system manufactures the Realizer's corpus**, the same bootstrap principle that builds the parser. Full treatment: docs/REALIZER.md §6.

### Q16. If no model reasons in text, where does "thinking" actually happen?

In **deterministic graph traversal** — the five-stage process: context pool accumulation over grammar/coreference/dependency edges (Stage 1), evidence-based neighborhood boundary evaluation (Stage 2), promotion of coherent neighborhoods to intent nodes with their knowledge pools attached (Stage 3), task derivation (Stage 4), and Response Graph assembly at checkpoints (Stage 5). Comparing two processes is *edge comparison over two pools*, not sentence generation. No thought-tokens are ever produced — unlike LLM chain-of-thought, reasoning never round-trips through natural language, which makes it both cheaper (zero decode steps) and fully inspectable (every inference traces to specific edges). Full treatment: docs/EXECUTION.md §2, docs/KNOWLEDGE.md §4.

### Q17. Can the Realizer "fix" an awkward or wrong Response Graph?

**No — by invariant.** The renderer may add function words, inflection, agreement, connectives realizing declared discourse relations, and ordering variation within frames — nothing else (docs/RESPONSE_GRAPH.md §5). A Realizer that "fixes" content is a Realizer that can *change* content, which is the hallucination vector reintroduced through the back door. If a graph is malformed, the correct behaviors are: constrained re-decode, escalation up the ladder, or surfacing the assembly error — never silent repair. Parse-back audits enforce this in production.

### Q18. Can OMEX models hallucinate at all?

The parser has no facts to invent — its output is structure over the given text, checkable span-by-span. The renderer emits only the graph it's handed, and parse-back audits verify it. Knowledge enters the system exclusively by reference, each with an address, bounded into pools by traversal. Failure modes still exist — a wrong parse, a wrong link — but they are **localized, detectable, and correctable** (a wrong link even self-signals via pool connectivity), which is categorically different from a fluent fabrication with no provenance.

### Q19. What is the CPU/GPU optimization story?

Ranked by impact: **(1)** non-autoregressive decoding — one pass per sentence vs. hundreds of decode steps; nothing else comes close; **(2)** batch sentences — extraction is embarrassingly parallel, and batching is what a bandwidth-bound small model needs; **(3)** everything resident — <1 GB int8, so no offloading logic should exist; **(4)** reasoning generates zero tokens — only the Realizer's short surface strings are autoregressive. CPU specifics: bandwidth math gives ~hundreds of sentence-passes/s single-stream on ordinary DDR5; int8/int4 (classification heads tolerate it well); SIMD integer kernels; parallelize across batch items, not within layers; NUMA-pin. GPU specifics: tiny models are launch-overhead-bound — CUDA Graphs, fused ops, big batches, device-resident inputs; transfer graphs, not activations. Orchestration: shared encoder amortizes; two-lane CPU/GPU pipeline (fabric traversal + ANN on CPU concurrent with encoder passes on GPU); cascade to a big model only on low-confidence outputs. Honest note: until OMEX models replace the LLM calls in a generation pipeline, the LLM dominates all cost — finishing that swap outranks every kernel trick.

### Q20. How does OMEX use an external knowledge fabric (like ZSEI), and what stays in memory?

**Always in RAM:** all OMEX weights (small, fixed), the fabric's structural index (mmap'd, O(1) hops), the linker's ANN index, and the hot caches. **On demand:** per-entry rich records and knowledge-pool neighborhoods — prefetched the moment the knowledge-span head fires so I/O overlaps compute. **Write-behind:** runtime-emitted graphs, inferred edges, and flywheel rendering pairs, batched per chunk. Batch a chunk's resolutions into one ANN pass + one grouped fetch; reuse the shared encoder's representations as span embeddings; cache on normalized span hashes with TTL invalidation. Structural payoff: because models hold no knowledge and per-query use is pool-bounded, **RAM scales with the working set, not the corpus** — the fabric can reach billions of entries while device memory stays flat. Full tiering: docs/DEPLOYMENT.md §2.

### Q21. Do I need Ozone-Studio or ZSEI to use OMEX?

No. They are the **reference implementations** of the things any host must provide: a validated graph pipeline (or existing conformant corpus) for training, a knowledge fabric with addressable paths + embeddings + **typed confidence-scored edges** for linking and bounded traversal, a Response Graph assembler, and a rendering-ladder floor. See docs/INTEGRATIONS.md §4 for the full requirements table.

### Q22. How is training data made, concretely?

Three streams: **(a)** a step-by-step LLM-driven pipeline over real text where every sentence, correction, section marker, modality span, grammar tree, and relationship is validated by repeated independent yes/no confirmation before acceptance — validation *is* the QA gate, with treebank calibration bounding factory drift; **(b)** synthetic generation via the master prompt in docs/TRAINING.md — one complete example per call (passage with controlled error injection, ordered sentence decomposition with corrections, full grammar trees over the complete taxonomy, cross-sentence relationships, coreference chains, knowledge references with paths, sections/paragraphs, candidate semantic units, a knowledge-graph fragment), scalable to hundreds of thousands via topic × register × error-density × seed; **(c)** the operational flywheel — validated ladder renderings from live systems. Every example trains parser **and** realizer (realizer on the corrected side only).

### Q23. How big are the models, and what hardware do they need?

Indicative: parser 100–300M, linker 50–150M, realizer 100–300M — the family under ~0.75 GB at int8 (final sizes are benchmark outcomes). Hardware: any modern phone, single-board computer, laptop, or server, CPU-only if need be; a GPU multiplies batch throughput but is not required.

### Q24. What about long documents — does OMEX do chunking or streaming?

Windowing long inputs is the **host's** job, done once on intake purely to fit context windows, parallelize, and anchor byte positions — it is deliberately **not** a format feature (adaptive chunking was removed from the design). OMEX models process whatever window they're handed; all downstream work is graph-based, with spans anchoring every node back to source bytes.

### Q25. What happened to the embedded optimizers / hardware profiles / memory budgets from earlier descriptions?

Removed, deliberately. They were over-engineering that didn't belong to what OMEX is: no execution-optimizer binaries, no hardware-aware annotations in the format, no per-node RAM budgets, no "ZSEI training-time optimization discovery." ZSEI's real role is knowledge fabric, not architecture analyzer. See "What OMEX Is Not" in the README.

### Q26. Are the performance numbers real benchmarks?

No — and the docs say so loudly. Everything quantitative is a **derivation** (parameter arithmetic, decoding-mode analysis, bandwidth math) with the steps shown, in docs/PERFORMANCE.md. Earlier circulated tables that presented projections as measured results are withdrawn as such. The benchmark program replaces derivations with measurements.

### Q27. Deterministic? Reproducible?

At temperature 0, structure prediction is deterministic; constrained decoding guarantees schema validity; spans make every output mechanically checkable against its source. Compare an LLM emitting JSON: stochastic, occasionally malformed, unverifiable.

### Q28. Which comes first on the roadmap?

Corpus generation → GrammarParser v0 (one language, structure + correction heads) → KnowledgeLinker v0 → Realizer v0 → the benchmark program (including the open design questions: per-language vs. multilingual, joint vs. separate heads, and OMEX-native vs. LLM-pipeline extraction on speed/quality/loops/energy) → per-language line-up and code/math models.

### Q29. Is "OMEX" still a fitting name?

Yes — it fits the corrected design better than the original one. **Omni**: across modalities (text, code, math, and the relational layers of perceptual media), across languages (one universal schema, per-language models), across scales (the same format from a phone-resident parser to a workstation family). **Execution**: prompt-first and executable end-to-end — a prompt enters, graphs flow through composed models, a response graph executes into text; the io contracts are literally execution interfaces. **Format**: the canonical representation of the model, its schemas, and the graphs that are the system's interchange currency. The redesign removed what wasn't OMEX (embedded optimizers, hardware profiles) and kept what always was: **a universal format for executing meaning.**
