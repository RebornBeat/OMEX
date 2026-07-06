Direct answer first, because you asked it twice and it's the seam everything else hangs on — then the README updated, then the eight new docs in full. The eight existing docs (CONCEPTS, ARCHITECTURE, FORMAT, TRAINING, EXECUTION, PERFORMANCE, INTEGRATIONS, FAQ) get their updates folded in and re-delivered whole in the continuation, exactly as you sanctioned — the list of what changes in each is at the end so nothing is ambiguous in the meantime.

---

# THE DIRECT ANSWER: Knowledge Paths, End to End

Your question decomposes into four, and each has a hard answer now.

**1. "We still have models *around* knowledge — did we truly capture all that?"** Yes, with one clarification that must be stated precisely: the KnowledgeLinker is a model *around* knowledge, but it **contains no knowledge either**. It contains a *mapping skill* — "span-in-context → address in the fabric." It knows how to find `/Metallurgy/HeatTreatment/Annealing`; it does not know what annealing is. When the fabric grows, the linker's index refreshes; its understanding of *metallurgy* never existed and never needs updating. So the full inventory of models is: parser (grammar skill), linker (addressing skill), realizer (rendering skill). Three skills, zero facts. That is the complete capture.

**2. "We aren't putting in all knowledge at once — but are we bringing in knowledge-specific models/params as needed?"** **No — and this is the Golden Rule that now gets its own document.** Weights are closed and static; knowledge is data. There are no medical parameters, no metallurgy adapters, no "load the physics head." The one exception is **structural adapters** (LoRA-class), which teach *syntax* — how to format a code block, a legal citation, a chemical formula — never facts. And here is the reframe that resolves your "even loading some beats LLMs" instinct: you are right that partial loading beats total loading — and OMEX *does* load only what's needed. It just does it **at the data tier instead of the weight tier**. Per query, a bounded fragment of the knowledge graph is fetched and injected as content. That is strictly superior to partial weight-loading, because data-tier loading is instant, current, provenance-carrying, and cannot be mis-remembered — while weight-tier loading, even partial, reintroduces the hallucination vector, the staleness problem, RAM creep, and per-domain model management. The instinct is satisfied; the mechanism is different.

**3. "Aggregated references can cover a huge array — unless we're traversing and creating context pools around *knowledge* as well?"** **Exactly — and yes, we are. This is the missing formalization, and it is now formalized.** A knowledge reference on a sentence node is an **entry point, not a payload**. The same Stage 1 / Stage 2 methodology that builds text neighborhoods applies to the knowledge graph — anchored by grammar instead of sentence adjacency. The query's own grammar graph tells the traversal *which edges matter*: "How does annealing **affect** steel?" gives the anchor `(Annealing, AFFECT, Steel)`, so the traversal from the `/Annealing` entry follows effect-class edges (`Reduces`, `Increases`, `Produces`, `Affects`) and scores paths that connect *both* anchors highest — it does not dump the entire `/Annealing` subtree. Expansion stops at a **relevance frontier** (the same semantic-frontier principle): when hops stop connecting to the co-anchors, when edge types drift off the anchor relation, when depth-decayed scores fall below threshold. The surviving subgraph is the **Knowledge Context Pool** for that AMT branch — bounded, scored, provenance-carrying. So "knowledge already aggregated on the branch" means precisely this pool, not the raw addresses. The full doc is `docs/KNOWLEDGE.md` below.

**4. "How can we rely on responses when we don't have a Realizer yet, or while it's still being trained?"** Because **the AMT never depends on the Realizer — it depends on the Response Graph *contract*, and three renderers satisfy that contract from day zero.** The rendering ladder: **Tier 0** — a deterministic template renderer (walks the skeleton, emits content in grammatical order using rule tables and the node properties; correct, plain, robotic; zero ML; can never fail to exist). **Tier 1** — the LLM rendering stand-in: the same model that serves as the graph factory renders the Response Graph under a render-only-the-graph instruction, and every output is parse-back validated (does its parse cover the graph's content nodes? did it add any ungrounded content span?). **Tier 2** — the OMEX Realizer, once trained, with escalation to Tier 1 on low confidence and Tier 0 as the unbreakable floor. And the flywheel: **every validated Tier 0/Tier 1 rendering is a `(response graph → text)` training pair** — operating the system *is* manufacturing the Realizer's corpus. The bootstrap principle you designed for the parser applies symmetrically to the realizer. One more thing your phrasing touched ("responses even if via internal thinking"): the AMT's *internal* thinking needs **no renderer at all**. Thinking is graph traversal; text exists only at user-facing checkpoints. Unlike an LLM's chain-of-thought, OMEX reasoning never round-trips through natural language — no thought-tokens are ever generated, which is both a correctness and an efficiency win.

And to close the loop on "have we captured the full knowledge paths": the path is now specified end to end with no underspecified segment — **Detect** (parser head) → **Resolve** (linker → address) → **Traverse** (bounded, grammar-anchored, Stage 1/2 over the fabric → Knowledge Context Pool) → **Inject** (pool facts become *filled* content nodes in the Response Graph, each carrying its address) → **Render** (the ladder). Previously the docs captured Detect and Resolve well; Traverse, Inject, and render-reliability were the underspecified segments. The three new documents below — KNOWLEDGE.md, RESPONSE_GRAPH.md, REALIZER.md — close them.

---

---

# FILE: README.md

---

# OMEX: Omni-Execution Format

**OMEX is a model format, a training methodology, and an execution interface for graph-native language models** — small, fast, auditable models that read and write explicit linguistic structure (grammar graphs) instead of hiding grammar, knowledge, and reasoning inside billions of opaque parameters.

If you have never encountered this project before, the one-paragraph version: today's large language models (LLMs) are enormous because they learn *everything at once* — grammar, style, world knowledge, reasoning — entangled across their weights. OMEX un-entangles them. It uses existing large models **once, as factories**, to produce hundreds of thousands of complete, validated *grammar graphs* — explicit structures describing exactly how every sentence is built, how sentences relate, what was grammatically wrong and how it was corrected, and which spans refer to knowledge topics. Small models are then trained directly on those graphs. The result is a family of models, each a fraction of a gigabyte, that natively **parse** text into structure, **correct** grammar, **reference** external knowledge without containing it, and **write** fluent text back out by running the same mapping in reverse — because writing is grammar traversal in reverse.

New to all of this? Start with **[docs/CONCEPTS.md](docs/CONCEPTS.md)**.

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
> OMEX models never contain world knowledge, and an OMEX system never loads "knowledge-specific" model parameters. The model family (Parser, Linker, Realizer) is structurally complete and static. When a task needs domain knowledge, that knowledge is fetched as **external graph data** from a knowledge fabric, bounded by grammar-anchored traversal, and injected as content into a **Response Graph**. The only dynamic parameters permitted are **structural adapters** (syntax formatting — code blocks, citations, notation), never facts.

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

---

---

# FILE: docs/KNOWLEDGE.md

---

# Knowledge in OMEX Systems: The Boundary, the Lifecycle, and Bounded Traversal

This document answers the questions every serious reader eventually asks: *if the models don't contain knowledge, where is it, how does it get selected, how much of it gets used per response, and how can the system answer anything at all?* The answers are mechanical, not hand-waved.

## 1. The Golden Rule: Weights vs. Data

**Grammar is a closed system → it lives in the weights.** A finite node taxonomy, a finite property set, compositional rules. Small models master it permanently.

**Knowledge is an open system → it lives as data.** Unbounded, growing, revisable, owned by an external knowledge fabric (a graph store of concepts, topics, relationships, addressable by path and searchable by embedding).

The rule, stated so it cannot drift:

> OMEX models never contain world knowledge, and an OMEX system never dynamically loads "knowledge-specific" model parameters. The model family is structurally complete and static. Domain knowledge is fetched as external graph data and injected as content into the Response Graph. The only dynamic parameters permitted are **structural adapters** (LoRA-class) that adjust *syntax rendering* — code-block formatting, legal-citation shape, mathematical notation — never facts.

### Why not even *partial* knowledge in the weights?

A reasonable instinct says: "a 2 GB domain model still beats a 40 GB general one — even loading *some* knowledge beats an LLM carrying topics it never needs." The instinct is directionally right, and OMEX honors it — **at the data tier, not the weight tier.** OMEX loads only the knowledge each query needs, per query, as data. That is strictly superior to loading it as parameters, because parameter-loading — even partial — reintroduces four failure modes:

1. **The hallucination vector returns.** The moment a model "knows" a fact, it can mis-recall that fact. A grammar-only Realizer mechanically *cannot* — it renders only the graph it is handed.
2. **The currency problem.** Facts change. Update the fabric and the very next response is correct. Facts in weights require retraining and redeployment.
3. **The RAM ceiling erodes.** "Just the top 100 domains" is the slippery slope back to multi-gigabyte models. The <1 GB family guarantee only holds if the answer is *zero* domains in the weights.
4. **Per-domain model management.** Infinite topics would demand infinite adapters, version skew between them, and a routing problem OMEX deliberately does not have.

### The linker is not an exception

The KnowledgeLinker is a model *about* knowledge, but it contains none: it holds a **mapping skill** — span-in-context → address. It can find `/Manufacturing/Metallurgy/HeatTreatment/Annealing`; it cannot state one fact about annealing. When the fabric grows, the linker's embedding index refreshes (cheap) or the linker fine-tunes (moderate); its nonexistent "knowledge of metallurgy" never needs updating.

## 2. The Knowledge Lifecycle: Five Steps, No Gaps

```
1. DETECT     Parser's knowledge-span head flags spans + coarse kind
              (concept | principle | procedure | entity | process | tool | material).
              The parser still does not know what the things ARE.

2. RESOLVE    KnowledgeLinker embeds each span in its sentence context,
              ANN-searches the fabric's entry embeddings, returns ranked
              addresses; low-margin candidates optionally zero-shot verified.

3. TRAVERSE   Bounded, grammar-anchored traversal FROM the resolved entries
              builds a Knowledge Context Pool (§3 — the heart of this doc).

4. INJECT     Facts from the pool become FILLED content nodes in the
              Response Graph, each carrying its fabric address as provenance.

5. RENDER     The rendering ladder (docs/REALIZER.md) turns the Response
              Graph into text. The renderer adds function words and
              inflection only — it can neither add nor alter facts.
```

Steps 1–2 were always specified. Steps 3–5 are the segments this document and its two siblings (RESPONSE_GRAPH.md, REALIZER.md) formalize.

## 3. Bounded Knowledge Traversal — the Knowledge Context Pool

**A knowledge reference is an entry point, not a payload.** `/HeatTreatment/Annealing` names a neighborhood that may contain hundreds of nodes: history, equipment, parameters, related processes, safety, standards. Dumping that subtree into a response would be context explosion — the exact failure OMEX exists to prevent. So the same Stage 1 / Stage 2 methodology that builds text neighborhoods applies to the **knowledge graph** — anchored by *grammar* instead of sentence adjacency.

### The algorithm

**Input:** the query/context grammar graph `G`; its resolved references `R`.

**Step 1 — Anchor extraction.** From `G`, extract the relational skeleton around the references: the governing verb(s), the co-arguments, the modifiers (conditions, temporal, purpose).

> *"How does annealing affect steel?"* → anchor = **(Annealing, AFFECT, Steel)**.

**Step 2 — Edge selection.** Map the anchor relation to fabric edge classes via a small deterministic table (zero-shot fallback for unmapped verbs):

| Anchor relation | Fabric edge classes followed |
|---|---|
| affect / influence / change | Affects, Increases, Reduces, Produces, HasProperty (shared) |
| compare / versus / difference | parallel property + effect edges on *both* entries |
| how-to / perform / steps | Procedure, Requires, Steps, Uses, Produces |
| why / cause / reason | CausedBy, Explains, TheoryOf, PrincipleOf |
| what-is / define | Definition, InstanceOf, PartOf, canonical description |
| depends / requires | DependsOn, Requires, Enables, Prevents |

**Step 3 — Scored expansion.** From each entry, walk the selected edge classes. Score every hop by: edge-class match to the anchor relation; **co-anchor connectivity** (paths connecting *both* Annealing *and* Steel score highest); stored edge confidence; hop-distance decay.

**Step 4 — The relevance frontier.** Stop expanding a branch when the evidence weakens: no remaining connection to a co-anchor; edge classes drifting off the anchor relation; depth-decayed score below threshold. This is the *semantic frontier* principle, applied to knowledge: the pool is "everything reachable **before the frontier**," never "everything reachable."

**Step 5 — The pool.** The surviving subgraph is the **Knowledge Context Pool** for this intent branch: nodes with their content, edges with their types, every element carrying its fabric address. Optionally, top facts at low margins get the fabric's zero-shot relevance verification.

### Worked example

Query grammar anchor: `(Annealing, AFFECT, Steel)`.

```
Entry: /HeatTreatment/Annealing
  ├─ Reduces ──────────► InternalStress      ✓ effect-class, and
  │                        └─ PropertyOf ──► Steel        ← co-anchor hit: TOP SCORE
  ├─ Increases ────────► Ductility
  │                        └─ PropertyOf ──► Steel        ← co-anchor hit: TOP SCORE
  ├─ Requires ─────────► ControlledCooling   ~ procedure-class: kept at depth 1,
  │                                            frontier stops expansion (no co-anchor link)
  ├─ InventedIn ───────► [history nodes]     ✗ off-anchor edge class: never followed
  └─ UsesEquipment ────► [furnace nodes]     ✗ off-anchor: never followed

Entry: /Materials/Steel
  ├─ HasProperty ──────► InternalStress      ✓ (joins the top-scoring path)
  └─ HasProperty ──────► Ductility           ✓
```

**Pool result:** `Annealing —Reduces→ InternalStress —PropertyOf→ Steel`; `Annealing —Increases→ Ductility —PropertyOf→ Steel`; one depth-1 context fact (`Requires ControlledCooling`). Perhaps six nodes and five edges — out of a neighborhood of hundreds. Bounded, relevant, provenance-carrying. **This pool — not the raw address — is what "knowledge already aggregated on the branch" means.**

### The Apple problem, solved at two layers

Ambiguity ("Apple": fruit or company?) is handled first at **resolution** — the linker embeds the span *with its sentence context*, so the verb "eat" pulls toward `/Biology/Fruit/Apple` and "invest" toward `/Business/Companies/Apple` — and second at **traversal**, because a wrong resolution produces a pool with near-zero co-anchor connectivity (the fruit entry has no scoring paths to "quarterly earnings"), which is a detectable signal that triggers re-resolution with the next-ranked candidate. Wrong links don't silently poison responses; they fail loudly at the pool stage.

## 4. Reasoning Over the Pool — Where "Thinking" Lives

The intelligence of an OMEX system lives in **deterministic traversal**, not in any model's weights:

- **Stage 1 over text** builds context pools along grammar/coreference/dependency edges.
- **Stage 1 over knowledge** (this doc, §3) builds knowledge pools along anchored fabric edges.
- **Stage 2** in both cases evaluates boundaries by accumulated evidence, never by lexical similarity.
- **Promotion** turns sufficient evidence into intent nodes — each carrying its text evidence *and* its knowledge pool.

Internal thinking **never generates text**. Comparing annealing and tempering is edge comparison over two pools, not sentence generation. Text appears only at user-facing checkpoints, via the Response Graph. This is a structural difference from LLM chain-of-thought: no thought-tokens, no natural-language round-trips inside reasoning, full inspectability of every inference step.

## 5. What the Knowledge Fabric Must Provide

Any fabric (the reference implementation is ZSEI — see docs/INTEGRATIONS.md) must offer: **addressable paths** (stable entry identities); **entry embeddings** (for the linker's ANN index); **typed relationship edges** with confidences (for anchored traversal); **fast structural hops** (traversal is the hot path); and ideally **zero-shot verification** hooks and **content references** (link, don't copy). Given those, the entire lifecycle above runs on it unchanged.

## 6. Summary Table: LLM Knowledge vs. OMEX Knowledge

| Property | LLM (knowledge in weights) | OMEX (knowledge as data) |
|---|---|---|
| Selection per query | Implicit, uncontrollable | Bounded pool via anchored traversal |
| Freshness | Frozen at training | Live — update fabric, next answer is current |
| Provenance | None | Every fact carries its address |
| Failure mode | Fluent fabrication | Loud, localized (empty/weak pool → detectable) |
| Memory scaling | With the corpus | **With the working set** — fabric grows to billions of entries, device RAM flat |
| "Load only what's needed" | Impossible | The default — per query, at the data tier |

---

---

# FILE: docs/RESPONSE_GRAPH.md

---

# The Response Graph

The Response Graph is the **handoff contract between reasoning and rendering** — the single structure through which everything the system has figured out becomes something a renderer can turn into text. It is a first-class citizen of the format, equal in standing to the input grammar graph. This document specifies what it is, where its content comes from, its schema, its assembly algorithm, and its invariants.

## 1. What It Is — and What It Is Not

The Response Graph is a grammar-schema skeleton whose content-bearing leaves are **pre-filled** with verified material: facts from Knowledge Context Pools, results from executed tasks, and decisions from intent traversal — each carrying provenance. It is *not* a prompt, *not* a summary, and *not* an instruction to a model to "answer the question." By the time a Response Graph exists, **the answer already exists**; the graph is the answer, structured. The renderer's only job is surface realization.

This is the mechanical reason the system cannot hallucinate at generation time: a renderer that may only add function words and inflection to pre-filled content has nothing to invent with.

## 2. Where Its Content Comes From

```
Intent branch reaches a checkpoint (task complete / clarification needed / answer ready)
        │
        ├── the branch's text evidence        (grammar-graph neighborhoods)
        ├── the branch's Knowledge Context Pool  (docs/KNOWLEDGE.md §3 — bounded facts + addresses)
        ├── executed task results             (verified outputs)
        └── voice properties                  (tone, formality, warmth, directness, humor)
        │
        ▼
   RESPONSE GRAPH ASSEMBLY  (deterministic, host-side)
        │
        ▼
   Rendering ladder (docs/REALIZER.md)
```

## 3. Schema

```json
{
  "schema_version": "1.0",
  "response_id": "resp-8842",
  "voice": {
    "tone": "informative",
    "formality": 0.6,
    "warmth": 0.55,
    "directness": 0.85,
    "humor": 0.05
  },
  "discourse": {
    "relations": [
      { "from_order": 1, "to_order": 2, "type": "Elaborates" },
      { "from_order": 2, "to_order": 3, "type": "Contrast"   }
    ]
  },
  "sentences": [
    {
      "order": 1,
      "sentence_type": "declarative",
      "granularity": "frame",
      "skeleton": {
        "node_type": "Sentence",
        "children": [
          { "node_type": "Subject",
            "content": "Annealing",
            "knowledge_ref": "/Manufacturing/Metallurgy/HeatTreatment/Annealing" },
          { "node_type": "Predicate",
            "children": [
              { "node_type": "MainVerb",
                "content": "reduces",
                "properties": { "tense": "present", "polarity": "affirmative" } },
              { "node_type": "DirectObject",
                "content": "internal stress",
                "knowledge_ref": "/Engineering/Materials/Mechanics/InternalStress" },
              { "node_type": "PrepositionalPhrase",
                "role": "location",
                "content": "in steel",
                "knowledge_ref": "/Materials/Steel" }
            ] }
        ]
      },
      "provenance": {
        "intent_branch": 412,
        "facts": [
          "edge:Annealing--Reduces-->InternalStress @0.97",
          "edge:InternalStress--PropertyOf-->Steel @0.93"
        ]
      }
    }
  ]
}
```

Notes on the schema:

- **`granularity`** is either `"frame"` (default — Subject/Verb/Object-level skeleton; the renderer chooses internal phrase structure, article placement, ordering variation) or `"tree"` (a fully specified grammar tree per docs/FORMAT.md, used when exact phrasing matters — legal text, quoted procedure steps, code). Frames give fluency headroom; trees give exactness. Both are legal per sentence, in the same response.
- **`discourse.relations`** carry the response's rhetorical structure between sentences (Elaborates, Contrast, Cause, Sequence, Example, Summary). The renderer uses them for connectives ("however," "because," "then") — connective *words* are surface realization; the *relations* are content, and they come from the graph.
- **`provenance`** is mandatory on every fact-bearing sentence: the intent branch it serves and the fabric edges/task results that ground it. Provenance is not rendered; it is audit metadata.
- **Modality content nodes** (code blocks, formulas) appear as typed content leaves (`node_type: "ModalityContent", modality: "code", content: "..."`), rendered verbatim inside the appropriate structural formatting (this is where structural adapters apply).

## 4. Assembly Algorithm (Deterministic, Host-Side)

1. **Select** the intent branch at its checkpoint; collect its Knowledge Context Pool and task results.
2. **Plan discourse.** Choose a rhetorical pattern from the checkpoint type and the pool's edge structure: *comparison* → parallel frames per entity with Contrast relations; *causal explanation* → cause-frame then effect-frame with Cause relations; *procedure* → ordered step frames with Sequence relations; *status/result* → result frame, then Elaborates frames for detail. Pattern selection is a deterministic table over relation types; genuinely novel shapes may use a zero-shot ordering call — ordering only, never content.
3. **Fill frames.** Map pool edges to frames mechanically: an edge `A —Reduces→ B` becomes Subject(A) / MainVerb(reduce, tense from context) / DirectObject(B); qualifiers become PrepositionalPhrases or Modifiers. Task results fill frames the same way (Subject = the artifact/action, Verb = its outcome verb, Object = its object).
4. **Attach provenance** per sentence; **attach voice** at the response level.
5. **Choose granularity** per sentence (frame by default; tree where exactness is required).
6. **Hand off** to the rendering ladder.

## 5. Invariants (Checkable by Any Consumer)

1. Every content-bearing leaf is pre-filled — the renderer never originates content.
2. Every fact-bearing sentence carries provenance (fabric addresses and/or task-result identifiers).
3. The renderer may add **function words, inflection, agreement, connectives realizing declared discourse relations, and ordering variation within a frame** — nothing else.
4. Parse-back must succeed: parsing the rendered text must recover every content leaf (coverage) and must surface **no content span absent from the graph** (zero-addition). This invariant is testable automatically and is enforced in training (cycle consistency) and in production audits.
5. Voice properties may alter word choice among function words and register — never facts.

## 6. Worked Example: A Comparison Response

Checkpoint: user asked to compare annealing and tempering for steel. The branch holds two knowledge pools.

Assembly plan: comparison pattern → sentence 1 (annealing effects), sentence 2 (tempering effects) with `Contrast`, sentence 3 (shared context: both are heat treatments) with `Summary`.

```
S1 frame: Subject(Annealing) Verb(reduces) Object(internal stress) PP(in steel)
          + coordinated second predicate: Verb(increases) Object(ductility)
S2 frame: Subject(Tempering) Verb(increases) Object(toughness) PP(in hardened steel)
          discourse: Contrast(S1→S2)
S3 frame: Subject(Both) Verb(are) Complement(controlled heat-treatment processes)
          discourse: Summary(S1,S2→S3)
```

Tier-2 rendering: *"Annealing reduces internal stress in steel and increases its ductility. Tempering, by contrast, increases toughness in hardened steel. Both are controlled heat-treatment processes."* Tier-0 rendering of the same graph: *"Annealing reduces internal stress in steel. Annealing increases ductility. Tempering increases toughness in hardened steel. Both are controlled heat-treatment processes."* — plainer, identical facts. **Content identity across renderers is the point:** facts are fixed at assembly; only fluency varies by tier.

---

---

# FILE: docs/REALIZER.md

---

# The Realizer: Graph → Text

The Realizer is the highest-risk component in OMEX, and this document treats it that way: the risk stated plainly, the architecture that addresses it, the fidelity mechanisms, the training curriculum, the **rendering ladder** that makes the system fully operational before the Realizer exists, and the evaluation program that decides whether it works.

## 1. The Risk, Stated Honestly

Text→graph is a well-solved problem class (neural parsers reach near-human agreement on structure at small sizes). Graph→text is harder. Known failure modes of graph-to-text generation: **telegraphic or robotic phrasing** (the model dutifully covers nodes and sounds like a list), **repetition**, **coverage gaps** (a node silently dropped), and — the one OMEX cannot tolerate — **additions** (fluent filler that introduces content not in the graph). Voice control adds a second axis: tone must alter register without touching facts. If the Realizer sounds stilted, users will not care how principled the backend is; if it adds content, the architecture's core guarantee breaks.

**Can it truly work?** Yes — with three grounds. First, precedent: data-to-text and meaning-representation-to-text systems demonstrate the task is learnable; their weaknesses are exactly the ones the mechanisms below target. Second, the asset no prior system had at this scale: **a corpus of exact inverse pairs** — every parser training example is a realizer training example reversed, hundreds of thousands of (graph ↔ fluent human text) pairs including register and voice variation. Third, the contract is *narrower* than general generation: the Realizer performs surface realization over pre-filled content, not open-ended composition.

## 2. The Fundamental Asymmetry

- **The Parser is non-autoregressive.** A sentence's structure is a static map; all arcs and labels can be predicted simultaneously in one forward pass.
- **The Realizer must be autoregressive.** Fluent text is sequential decision-making: the word chosen at step *N* constrains the grammatical options at step *N+1*. Non-autoregressive text generation of this kind produces disfluent output. This asymmetry is accepted, not fought — and it is cheap in context: the Realizer generates only final, user-facing sentences (typically tens of tokens per checkpoint), never thought-tokens, never structure-as-JSON.

**Consequence for the family:** the Realizer is the slowest OMEX model per invocation and still orders of magnitude cheaper than LLM generation, because it decodes short surface strings from a small model, once per checkpoint.

## 3. Architecture

**Graph encoder → autoregressive transformer decoder, with voice prefix and coverage tracking.**

1. **Graph encoding.** A graph transformer (or GNN) encodes the Response Graph: every node (type, content, properties, discourse role) becomes an embedding; structural relations become attention biases. Frame-granularity and tree-granularity inputs share this encoder.
2. **Voice conditioning.** The voice vector (tone, formality, warmth, directness, humor) is embedded and **prepended as a prefix** to the decoder's context — the model learns that the same graph renders differently under different prefixes, at the level of register and function-word choice only (enforced by training data construction and cycle consistency, §4).
3. **Autoregressive decoding with graph attention.** The decoder generates tokens attending jointly to its own prefix and the graph embeddings.
4. **Coverage tracking.** A coverage vector accumulates attention mass per content node; decoding is steered (and training penalized) toward covering every content node exactly once — the classical remedy for omission and repetition.
5. **Constrained decoding.** The decoder operates under the grammar schema: agreement and inflection consistent with node properties (a `MainVerb {tense: past}` must surface in past tense); tree-granularity sentences additionally constrain phrase order.

### The three candidate paths (and the chosen one)

| Path | Design | Pros | Cons |
|---|---|---|---|
| **A** | Pure autoregressive decoder conditioned on a linearized graph | Maximum fluency; simplest | Highest addition risk; weakest structural grounding |
| **B** | Deterministic templates + small neural smoothing (inflection, agreement) | 100% fidelity by construction; trivial to verify | Robotic and repetitive over long responses |
| **C** | **Graph-encoder + autoregressive decoder + coverage + cycle consistency (the OMEX way)** | Fluency from autoregression; fidelity from coverage + parse-back; voice-controllable | Heaviest to train; slowest of the family |

Path B is not discarded — it **is Tier 0 of the rendering ladder** (§6). Path C is the target; Path A is a training-ablation baseline, not a product.

## 4. Fidelity Mechanisms

1. **Coverage loss** — penalize under-attended (omitted) and over-attended (repeated) content nodes.
2. **Cycle-consistency loss** — during training, parse the generated text with the (frozen) GrammarParser and penalize any divergence from the input Response Graph's content set: missing leaves (coverage failure) and, with the heaviest penalty, **content spans not present in the graph** (additions). This turns the parser into the Realizer's permanent adversarial auditor — the same instrument used in production audits.
3. **Property enforcement** — decoding masks that make inflection agree with node properties.
4. **Register-isolation in data** — voice-variant training pairs are constructed so the *content leaves are identical across variants*; only surface register differs. The model therefore cannot learn to express "warmth" by adding claims.

## 5. Training

**Data.** (a) The parser corpus reversed — with one asymmetry: the parser's input side includes erroneous originals, but **the Realizer trains only on the corrected side** (graph → corrected sentence); it must never learn to emit errors. (b) Synthetic voice variants: the same graph rendered at multiple voice settings by the generation prompt (docs/TRAINING.md), content-leaf-identical by construction. (c) The operational flywheel (§6): every validated Tier-0/Tier-1 rendering in a live system is a new training pair.

**Curriculum.** Structure-only single frames → property-rich frames (tense/aspect/polarity/number) → tree-granularity exact rendering → multi-sentence discourse with relations and connectives → voice conditioning → modality content embedding (code/math leaves rendered verbatim within prose) → long mixed responses.

## 6. The Rendering Ladder — Operational from Day Zero

The AMT and the orchestrator depend on the **Response Graph contract**, not on the Realizer's existence. Three renderers satisfy the contract:

- **Tier 0 — Template renderer.** Deterministic: walks the skeleton, emits content in grammatical order using per-language rule tables (constituent order, agreement from properties, default connectives per discourse relation). Correct, plain, robotic. Zero ML. Exists on day zero; can never be unavailable. *(This is Path B, kept as the floor.)*
- **Tier 1 — LLM rendering stand-in.** The same model that serves as the graph factory receives the Response Graph with the instruction: *render into fluent text; you may add function words, inflection, and connectives realizing the declared discourse relations; you may not add, remove, or alter any content.* Every output is **parse-back validated** (coverage + zero-addition, per RESPONSE_GRAPH.md invariant 4); failures are re-rendered or dropped to Tier 0.
- **Tier 2 — OMEX Realizer.** Swaps in when trained; escalates to Tier 1 on low decoder confidence or parse-back failure; Tier 0 remains the unbreakable floor.

**The flywheel:** every validated Tier-0/Tier-1 rendering is a `(response graph → text)` pair. **Operating the system manufactures the Realizer's corpus.** The bootstrap principle that builds the parser builds the Realizer symmetrically — and this is the complete answer to "how can we rely on responses before the Realizer exists": responses never waited for it; content is fixed at assembly time and identical across tiers; only fluency improves as the ladder climbs.

## 7. Evaluation

- **Parse-back fidelity:** content-leaf coverage rate (target: ~100%) and **addition rate (target: ~0, hard gate)** — measured automatically at scale.
- **Fluency:** human ratings vs. Tier-1 renderings of the same graphs (the fair baseline: same content, different renderer).
- **Voice fidelity:** classifier + human checks that voice settings shift register without shifting content.
- **Robustness:** adversarial graphs (sparse frames, dense trees, mixed modality leaves, contradiction-shaped discourse) — the Realizer must render what it is given or abstain to Tier 1, never "fix" content.
- **Latency/energy:** tokens/s and J/response on target devices, alongside the family.

## 8. Failure Modes and Responses

| Failure | Detection | Response |
|---|---|---|
| Omission | Parse-back coverage < 100% | Re-decode with boosted coverage; else Tier 1 |
| Addition | Parse-back finds ungrounded span | **Hard reject**; Tier 1; log for training |
| Stiltedness | Human/fluency metric drift | More voice-variant data; decoder fine-tune; acceptable interim — facts unaffected |
| Property violation (wrong tense etc.) | Deterministic check vs. node properties | Constrained re-decode |
| Confidence collapse on novel shapes | Decoder entropy spike | Escalate to Tier 1; add shape to curriculum |

**Verdict.** The Realizer is buildable: precedent exists, the inverse corpus is uniquely strong, the contract is narrow, the auditor (the parser) is free, and the ladder means the system never depends on it prematurely. It will be the most-iterated model in the family — and the architecture is designed so that iterating it risks nothing but fluency.

---

---

# FILE: docs/MODALITIES.md

---

# OMEX Across Modalities

The OMEX thesis — *closed structural systems can be mastered natively by small models; open content stays external* — is not text-specific. This document maps the thesis onto each modality: where it applies fully, where it applies partially, and where it honestly does not.

## 1. The Applicability Test

Ask of any modality: **is its structure a closed system?** A finite node taxonomy, finite relation set, compositional rules → full OMEX applicability. Structure that only exists *after* an open-world perception step → partial applicability (the relational layer only).

| Modality | Structure closed? | OMEX applicability |
|---|---|---|
| Text | Yes (grammar) | Full — the reference case |
| Code | **Strictly** yes (formal grammar) | Full — purest case |
| Math | **Strictly** yes (formal derivation) | Full |
| Image | No (perception is open) / relational layer yes | Relational layer only |
| Audio | Same split | Relational/prosodic layer only |
| Video | Same split | Relational/temporal layer only |

## 2. Code-OMEX (the purest application)

Code's grammar is stricter than natural language: the AST *is* a formal closed system.

- **Code Parser:** source → AST **plus** a semantic edge layer over it: `Calls/CalledBy, Imports/ImportedBy, Defines/DefinedBy, DependsOn/RequiredBy, DataFlows, ControlFlows, ImplementsPattern, UsesPattern, IntentSolves, ExtendsAbstraction` — non-autoregressive, single pass per unit, spans anchoring every node to source bytes.
- **Correction analog:** lint/fix pairs play the role grammar corrections play in text — original/fixed with typed, localized edits.
- **Knowledge spans:** API names, package references, protocol identifiers → detected by a head, resolved by the linker to fabric addresses (package registries, API documentation entries). The model knows *that* `tokio::spawn` is a knowledge-bearing reference; the fabric knows *what* it is.
- **Code Realizer:** response graph (AST skeleton + filled identifiers/content) → source text. **Easier than the text Realizer**: rendering is nearly deterministic because code syntax admits almost no register variation — a `FunctionDef` node has essentially one surface form per language. Structural adapters carry per-language formatting conventions.
- **Native understanding claim:** execution-path and dependency structure are read off the graph, not statistically guessed — a Code-OMEX system *knows* what calls what because the edges say so, with spans to prove it.

## 3. Math-OMEX

- **Parser:** expressions/proofs → derivation graphs: `LogicallyImplies, LogicallyEquivalent, Contradicts, Generalizes, SpecialCaseOf, UsedToProve, DischargesAssumption, AssumesIn, DependsOn`, with per-step justification slots.
- **Knowledge spans:** named theorems, constants, definitions → fabric addresses; the model never contains the theorem, only the recognition that one is referenced.
- **Realizer:** derivation graph → notation and prose; tree-granularity dominates (exactness matters); structural adapters carry notation conventions (LaTeX, plain text).
- **Native understanding claim:** proof structure — what follows from what, under which assumptions — is explicit and auditable step by step.

## 4. Perceptual Modalities: the Honest Boundary

**OMEX does not do perception.** Pixels→objects and waveforms→events are open-world learning problems requiring perceptual models of real size — outside the closed-system thesis, and no amount of graph design changes that.

**What OMEX does:** once a perception front-end emits entities/events, an **OMEX-Relational model** builds the typed relational graph over them:

- **Image:** `SpatialAbove/Below/Left/Right, Occludes, PhysicallySupports, ContainedIn, PartOf/HasPart, FocalSubject, LeadsEyeTo`, plus affordances (`CanSupport`, `CanContain`).
- **Audio:** `Precedes/Follows, Causes, OverlapsWith, InterruptedBy, ResponseTo, ToneIndicates, PitchCorrelates`, plus prosody structure (pitch, rate, energy, emotional markers).
- **Video:** `InteractsWith, CollidesWith, FollowsSpatial, CausesEvent, NarrativeLeadsTo, AudioSyncsTo`, plus timed interactions (`Touches, Approaches, Leaves, PassesBy` with start/end times).

**Worked example.** Front-end output: `[dog, couch, ball]` with boxes and timestamps. OMEX-Relational output: `dog —On→ couch; ball —LeftOf→ couch; dog —LooksAt→ ball` (+ temporal edges in video). Downstream, these graphs join the same intent-building traversal as text graphs.

**Pipeline shape:** `perception front-end → OMEX-Relational → cross-modal stitching`. Code and math need no front-end at all.

## 5. Cross-Modal Stitching

Modality graphs reference one another via the cross-modal edge set: `DescribedBy/Describes, ImplementedIn/Implements, VisualizedAs/Visualizes, SyncedTo/SyncedBy, ReferencedBy/References`. When a task spans text and code, each model emits its graph and the host stitches them: a text sentence node `—Describes→` a code function node. **Execution contexts expand through graph stitching; parameters never merge** (see ARCHITECTURE.md §7). Response Graphs may carry modality content leaves (a code block inside a prose answer), rendered verbatim under structural-adapter formatting.

## 6. Schema Registry and Roadmap

Every modality's node/edge sets live in the versioned `schema/edge_schema.json` registry of its models; composition legality is schema-version compatibility, identical to text. Build order: **text → code → math** (closed systems, corpus-manufacturable by the same factory pattern) → **relational models for image/audio/video** (dependent on chosen perception front-ends and entity-graph corpora).

---

---

# FILE: docs/LANGUAGES.md

---

# Language Strategy

## 1. The Default: Universal Schema, Per-Language Models

**One schema, many models.** The grammatical node taxonomy and property set are language-general — the same `Subject`, `RelativeClause`, `{tense, case, gender…}` vocabulary describes every language's realization (in the spirit of universal dependency annotation). The **models** are per-language: a compact parser/realizer pair per language, selected by a tiny language-ID router. Downstream consumers (intent building, knowledge traversal, response assembly) see one schema regardless of language — they never know or care which parser ran.

**Why per-language (at equal size):** better accuracy than multilingual peers except in low-resource transfer; smaller working sets (load only the languages a device uses — the local-first principle applied to languages); independent update cadence per language; and no cross-language interference in the correction head (error distributions are strongly language-specific).

## 2. Alignment with Existing Annotation Standards

The OMEX schema **maps onto Universal Dependencies-style annotation** wherever a correspondence exists (a documented mapping table ships with the schema). Two payoffs: existing human-annotated multilingual treebanks become **calibration sets** for the graph factory (the ≥95% agreement gate in docs/RISKS.md is measured against them), and they seed **bootstrap corpora** for languages where LLM factory quality is initially weaker. Where OMEX's taxonomy is richer than the standard (correction pairs, knowledge spans, modality spans, discourse relations), those layers are OMEX-native and generated by the factory pipeline.

## 3. The Multilingual Bootstrap Path (Low-Resource Languages)

1. Train a **multilingual bootstrap parser** on high-resource languages + available treebanks.
2. Use it (plus the LLM factory with `LANGUAGE` set) to generate validated graphs for the low-resource language.
3. **Distill a per-language model** from the accumulated corpus once volume suffices.
4. Retire the multilingual model for that language; keep it as the router's fallback for未covered languages.

Multilingual models are a **means**, per-language models the **end** — except where measurements say otherwise (§5).

## 4. Corpus Multiplication Management

Per-language lines multiply corpus cost. Containment: the master generation prompt is language-parameterized (one generator, N languages); curricula are shared (clean → complex → error-injected → mixed-register → knowledge-dense) with language-specific error-type distributions; treebank calibration bounds factory drift per language; and language rollout is demand-ordered, not exhaustive — start with one language, prove the loop, expand.

## 5. The Open Question, Stated Plainly

Per-language vs. all-languages-together is **explicitly unsettled** and is a benchmark line item (docs/BENCHMARKS.md): equal-parameter comparisons on parse accuracy, correction quality, and realizer fluency, per language tier (high/mid/low-resource). The default above is the recommendation with reasons; **measurements decide.** Everything else in this document is invariant under either outcome — the universal schema guarantees that.

---

---

# FILE: docs/RISKS.md

---

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

---

---

# FILE: docs/BENCHMARKS.md

---

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

---

---

# FILE: docs/DEPLOYMENT.md

---

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

---

---

# WHAT CHANGES IN THE EXISTING EIGHT (delivered in full in the continuation)

So nothing is ambiguous while the continuation lands, these are the exact updates being folded in — each doc re-delivered whole:

- **CONCEPTS.md** — new §: *The Golden Rule: Weights vs. Data* (newcomer framing); §6 extended with the entry-point-not-payload principle and a pointer to KNOWLEDGE.md; glossary additions (Knowledge Context Pool, Response Graph, rendering ladder, cycle consistency, parse-back).
- **ARCHITECTURE.md** — new §: *The Boundary: Weights vs. Knowledge Data* (normative statement); Realizer section updated to the asymmetry + Path C summary with pointer to REALIZER.md; contracts section notes the Response Graph as a first-class contract.
- **FORMAT.md** — Response Graph added to the interchange section (schema summary + pointer to RESPONSE_GRAPH.md); frame vs. tree granularity; discourse relations registered in `edge_schema.json`.
- **TRAINING.md** — Realizer curriculum subsection (corrected-side-only rule, voice-variant construction with content-leaf identity, cycle-consistency training); the operational flywheel as corpus source (c); treebank calibration gate cross-referenced.
- **EXECUTION.md** — runtime loop expanded with the bounded knowledge traversal step and Response-Graph assembly step; new §: *Reasoning Without Text* (AMT-style internal thinking: Stage 1/2/promotion, no thought-tokens); new §: *Operating Before the Realizer Exists* (the rendering ladder inline); Realizer's autoregressive cost noted honestly in the speed section.
- **PERFORMANCE.md** — Realizer decode cost added to the derivations (short autoregressive surface strings per checkpoint vs. LLM full-generation, arithmetic shown); explicit note that BENCHMARKS.md supersedes this doc result-by-result.
- **INTEGRATIONS.md** — AMT primer expanded with the five-stage thinking process and the knowledge-pool aggregation on branches; host requirements table gains "typed fabric edges with confidences" and "response-graph assembler" rows; memory strategy cross-referenced to DEPLOYMENT.md.
- **FAQ.md** — new questions: *Do you load knowledge-specific models or parameters as needed?* (No — Golden Rule, data-tier loading reframe); *A knowledge reference covers a huge topic — how is the response bounded?* (Knowledge Context Pools, anchored traversal, relevance frontier); *How does the system respond before the Realizer exists?* (the rendering ladder + flywheel); *Where does "thinking" happen if no model reasons in text?* (deterministic traversal, no thought-tokens); *Can the Realizer fix a bad graph?* (No — invariant 3; it renders or abstains).

Say "continue" and the next response delivers those eight, whole, with these updates integrated — same rules: complete, inline, nothing dropped.

Continuing exactly where the last response ended: the eight existing docs, re-delivered whole and complete with every promised update integrated — nothing from the prior versions dropped, all new material folded in. With these, the full set is complete: README + 16 docs.

---

---

# FILE: docs/CONCEPTS.md

---

# OMEX Concepts — Background from Zero

This document assumes no prior knowledge. It explains the three eras of language processing, what a grammar graph actually is, the central design principle of OMEX — **grammar is a closed system; knowledge is an open one; they should never live in the same weights** — and the rule that keeps the whole architecture from ever drifting back.

## 1. What Traditional NLP Was

The distinction to hold onto: **NLP** (Natural Language Processing) is the *field*; **SLMs** and **LLMs** are specific kinds of *models* used within that field.

Historically, NLP systems separated language into components:

```
Sentence
    ↓
Tokenizer                (split text into units)
    ↓
Part-of-Speech Tagger    (label each word: noun, verb, ...)
    ↓
Parser                   (build the structural tree)
    ↓
Grammar Rules            (hand-written constraints)
    ↓
Semantic Analysis        (attempt meaning)
    ↓
Task
```

For example, given:

> "The cat chased the mouse."

an NLP pipeline might produce:

```
The      Determiner
cat      Noun
chased   Verb (Past)
the      Determiner
mouse    Noun

Subject = cat
Verb    = chased
Object  = mouse
```

Every stage was often a different model or a collection of handcrafted rules. **Grammar was something engineers explicitly encoded.** This gave precision and auditability — and brittleness. The rules were a dumb layer: they could not adapt, could not handle the endless irregularity of real language, and the field plateaued.

| Aspect | Traditional NLP | SLM (Small Language Model) | LLM (Large Language Model) |
|---|---|---|---|
| Primary approach | Hand-crafted rules, statistical models, feature engineering | Neural transformer trained on language | Large transformer trained on enormous corpora |
| Grammar understanding | Explicit rules or learned statistical patterns | Learns grammar implicitly | Learns grammar implicitly, at much larger scale |
| Vocabulary | Usually task-specific | Broad | Extremely broad |
| Context length | Limited | Moderate | Large |
| Generalization | Usually narrow | Good | Excellent |
| Need for linguistic rules | Often yes | No | No |

## 2. What SLMs and LLMs Actually Learn

Modern language models contain **no grammar engine**. They learn grammar from exposure to massive amounts of text. During training they repeatedly see examples like:

```
I am going
You are going
He is going
```

and eventually internalize the pattern:

```
Pronoun → verb agreement → correct sentence
```

No one ever tells the model:

```
if subject == "he": use "is"
```

Instead, the model's parameters gradually encode the relationship. And they learn many levels simultaneously:

```
Characters → Subwords → Words → Grammar → Meaning → Reasoning patterns → Writing style
```

There is **no dedicated grammar module**. Grammar becomes distributed across millions or billions of learned parameters.

A concrete demonstration. Given the sentence:

> "The dogs that live on the hill are barking."

Traditional NLP would explicitly identify: *dogs* = plural noun; *that live on the hill* = relative clause; *are* = plural verb; *barking* = present participle.

An LLM instead predicts the next token from everything before it:

```
The dogs that live on the hill ___

are      98%
is        1%
were      0.5%
was       0.3%
```

It has learned that **"dogs"** is the grammatical subject — even though **"hill"** is the closest noun. Does the LLM "know" grammar? In practice, yes — but not as explicit rules. Researchers describe this as an **emergent internal representation** of grammar: syntax is recoverably *in there*, entangled with everything else.

The SLM/LLM difference is scale, not mechanism:

```
SLM  (roughly 1B–10B parameters)         LLM  (roughly 50B–1T+ parameters)
✓ grammar   ✓ syntax   ✓ semantics       ✓ all of that, plus:
limited world knowledge                  ✓ discourse  ✓ style
limited reasoning                        ✓ multilingual patterns
smaller context                          ✓ stronger reasoning
```

## 3. The Historical Insight That Makes OMEX Possible

Before SLMs/LLMs, moving away from hardcoded rules was impossible — the rules were all we had, and they were a dumb layer. Now the situation is inverted, and something new becomes possible:

**SLMs/LLMs can manufacture what hand-coding never could: hundreds of thousands of complete, semantically-annotated grammar graphs.** Full structural trees for every sentence. Cross-sentence relationships. Coreference chains. Before-and-after grammar corrections. References from text spans to knowledge topics. All validated.

And then the revolutionary step: **train small models directly on those graphs.** The result surpasses traditional NLP — because the components are no longer hand-coded rules but neural models with natural semantic understanding — while discarding the LLM's defining cost, because grammar without embedded knowledge is *small*.

OMEX expands off of NLP **and** SLMs/LLMs. It would not be possible without either.

## 4. What a Grammar Graph Is

A grammar graph is an explicit structural representation of language. The crucial property: sentences are **not** flat word chains.

> "I want to try on a suit I saw in a shop that's across the street from the hotel."

is *not* really:

```
I → want → to → try → on → a → suit → ...
```

It is closer to:

```
Sentence
│
└── Predicate (want)
    ├── Subject
    │   └── I
    └── Infinitive Clause
        └── try
            ├── Object
            │   └── suit
            └── Relative Clause
                └── saw
                    ├── Subject
                    │   └── I
                    ├── Object
                    │   └── suit
                    └── Location
                        └── shop
                            └── Relative Clause
                                └── across
                                    ├── street
                                    └── hotel
```

Notice something important: **nothing is duplicated.** Everything simply **attaches to its grammatical parent.** Every node has exactly one parent and any number of children. Everything belongs somewhere. There is no clutter.

A second example:

> "The dog quickly chased the cat through the garden."

```
Sentence
│
└── MainClause
    ├── Subject
    │   └── NounPhrase
    │       ├── Determiner  "The"
    │       └── Noun        "dog"
    └── Predicate
        ├── Verb            "chased"
        ├── Adverb          "quickly"
        ├── DirectObject
        │   └── NounPhrase
        │       ├── Determiner  "the"
        │       └── Noun        "cat"
        └── PrepositionalPhrase
            ├── Preposition          "through"
            └── ObjectOfPreposition  "garden"
```

### Two layers, cleanly separated

```
Document Layer                          Grammar Layer (one tree per sentence)
──────────────────────────              ─────────────────────────────────────
Document                                Sentence
└── Section                             ├── Clause
    └── Paragraph                       │   ├── Subject
        └── Sentence  ─────────────────►│   ├── Predicate
                                        │   │   ├── Verb
                                        │   │   ├── Object
                                        │   │   ├── Complement
                                        │   │   └── Modifiers
                                        │   └── Nested Clauses
                                        └── Punctuation (optional)
```

The document layer answers *"where did this come from?"* — pure containment. The grammar layer answers *"how is this built?"* — one tree per sentence, grammar staying **local to each sentence**, with explicit relationship edges connecting sentences to each other and paragraphs/sections to their parents.

### Node properties

Structure says what a node *is*; **properties** say *"describe me"*:

```
GrammarProperties {
    tense, aspect, mood, voice,
    person, number, gender, case,
    definiteness, comparison, polarity, subtype
}
```

So a verb node might carry `{ tense: Past, aspect: Perfect, mood: Indicative, voice: Active }`; a noun node `{ number: Plural, case: Possessive }`; a pronoun `{ subtype: Personal, person: First, number: Singular, case: Subjective }`.

### Beyond the single sentence

A complete graph corpus also carries:

- **Cross-sentence relationships** — typed edges between sentences: *Elaborates, Causes, Enables, Prevents, Contradicts, Exemplifies, Summarizes, TemporalPrecedes, PartOf, SimilarTo*.
- **Coreference chains** — "John Smith → He → the manager" tracked as one entity across sentences, each mention tagged with its grammatical role.
- **Correction pairs** — every sentence stored as *original* (exactly as written, errors included) and *corrected*, with each edit localized to a span and typed (agreement, tense, article, spelling, punctuation, word order, fragment, run-on).
- **Structural anchoring** — every node carries character offsets back into the source text, so any claim about the text is mechanically checkable.

## 5. Traversal: What Graphs Are *For*

A grammar graph is not decoration — every edge type carries a deterministic traversal meaning. When a system walks the graph to understand a sentence, the verb is merely the **execution anchor**; the real traversal follows typed edges outward:

| Grammar relationship | Traversal invariant | Effect on the accumulated context |
|---|---|---|
| Subject → Predicate | Establish actor | Include the executor of the action |
| Predicate → Verb | Establish execution root | Anchor the traversal |
| Verb → Direct Object | Expand primary target | Include what the action operates on |
| Verb → Indirect Object | Expand recipient | Include who receives the result |
| Verb → Prepositional Phrase | Expand contextual constraints | Include instruments, locations, relations |
| Verb → Adverbial | Expand execution modifiers | Include manner, degree, frequency |
| Verb → Purpose Clause | Expand downstream objective | Include intended outcome |
| Verb → Conditional Clause | Expand execution conditions | Include prerequisite branches |
| Verb → Temporal Clause | Expand execution ordering | Include sequencing information |
| Relative Clause → Modified Node | Expand defining properties | Refine the identity of the referenced node |

So "understanding" becomes a mechanical procedure: visit, expand along typed edges, accumulate a context pool that represents the complete executable meaning of the sentence — no guessing.

## 6. Grammar vs. Knowledge — The Central Separation

**Grammar is a closed system.** A finite taxonomy of node types (about 170, from *Sentence* through *Punctuation* — see docs/FORMAT.md), a finite property set, compositional rules. Closed systems can be mastered by small models.

**Knowledge is an open system.** Unbounded, growing, revisable. Embedding it in weights is *why* LLMs are enormous — and why they hallucinate: knowledge in weights has no provenance.

OMEX models learn the closed system natively and treat the open system as an **address space**. A model does not need to know the history of the Roman Empire to parse a sentence that mentions it. What it *does* need — and is trained for, exactly like grammar correction — is to recognize **when a sentence or paragraph revolves around a knowledgeable topic**, and emit a **reference** to the applicable path in an external knowledge graph.

Example. Given:

> "The annealing process relieves internal stress."

an OMEX parser maps the grammar perfectly, and additionally flags *annealing* and *internal stress* as knowledge-bearing spans. A companion model (the KnowledgeLinker) resolves those spans against an external knowledge fabric, returning exact addresses like `/Manufacturing/Metallurgy/HeatTreatment/Annealing`. The knowledge stays external; the reference travels with the graph.

**And crucially: a reference is an entry point, not a payload.** The address `/HeatTreatment/Annealing` names a neighborhood that may hold hundreds of facts — history, equipment, parameters, safety, standards. The system never dumps that neighborhood into a response. Instead, the *query's own grammar* anchors a **bounded traversal** of the knowledge graph: only the facts connected to what was actually asked survive into a small, scored **Knowledge Context Pool**. Because downstream meaning-building accumulates neighborhoods over these same graphs, every branch of the eventual intent structure arrives with its applicable knowledge pool **already aggregated** — no separate knowledge pass, no loading everything into memory. The full mechanism is specified in **docs/KNOWLEDGE.md**.

## 7. The Golden Rule: Weights vs. Data

The separation in §6 only holds if one rule is never broken:

> **Weights are closed. Knowledge is data.** OMEX models never contain world knowledge, and an OMEX system never dynamically loads "knowledge-specific" model parameters. There is no medical version of the parser, no metallurgy adapter for the realizer, no "load the physics head." Domain knowledge is always fetched as external graph data, per query, bounded by traversal, and injected as content. The only dynamic parameters permitted are **structural adapters** — small add-ons that teach *syntax formatting* (code blocks, legal citations, mathematical notation), never facts.

A newcomer's natural objection: *"But even loading some knowledge into a small model would still beat a giant LLM that carries everything!"* The instinct is right — and OMEX honors it, **at the data tier instead of the weight tier**. Per query, only the needed fragment of the knowledge graph is loaded — instantly, currently, with provenance. Loading knowledge as *parameters*, even partially, would reintroduce everything OMEX exists to eliminate: a model that "knows" a fact can *mis-remember* it (the hallucination vector returns); facts in weights go stale until retraining (the currency problem); "just the top 100 domains" is the slippery slope back to multi-gigabyte models (the RAM ceiling erodes); and infinite topics would demand infinite adapters (unmanageable). Data-tier loading gets the benefit without any of the costs.

One subtlety worth naming: the KnowledgeLinker is a model *about* knowledge, yet contains none. It holds a **mapping skill** — from a text span in context to an address in the fabric. It can find where "annealing" lives; it cannot state one fact about annealing. Three skills across the family — grammar, addressing, rendering — and zero facts.

## 8. Writing Is Grammar Traversal in Reverse

Everyone assumes language works like: *Text → Grammar → Meaning → Response.* That skips the most important layer. The real loop:

```
Input Text
  ↓
Grammar Graph        (parse)
  ↓
Intent Graph         (understand — traversal over grammar)
  ↓
Knowledge Pools      (bounded traversal of the external fabric)
  ↓
Decision             (reason / execute tasks — no text is generated here)
  ↓
Response Graph       (plan — verified content, structure, tone, provenance)
  ↓
Grammar Graph        (structure the response)
  ↓
Output Text          (render)
```

**Grammar exists on both sides. It is both the parser AND the renderer.** Generation is not token-by-token guessing; it is rendering a verified structure. A realizer trained on the inverse of the parser's data renders only what the response graph contains — which is why it cannot introduce facts that were never handed to it. And because the Response Graph is a contract rather than a model, the system can render responses from day zero through a **rendering ladder** — a deterministic template renderer and a validated LLM stand-in serve until the trained Realizer takes over (docs/REALIZER.md). Content is identical across all rungs; only fluency climbs.

## 9. Glossary

- **Token** — the unit LLMs read/write (a word-piece). **Autoregressive decoding** — producing output one token at a time, each step depending on the last; this is why LLM generation is inherently sequential and slow. **Non-autoregressive** — predicting an entire structure in one forward pass.
- **Zero-shot** — asking a model to perform a task from instructions alone, with no task-specific training.
- **Forward pass** — one full run of input through a model.
- **Quantization (int8/int4)** — storing weights in 8 or 4 bits instead of 16/32, shrinking memory and bandwidth needs.
- **Knowledge fabric / knowledge graph** — an external store of concepts, topics, and their relationships, addressable by path and searchable by meaning.
- **Coreference** — different expressions ("John", "he", "the manager") referring to the same entity.
- **Span** — a start/end character range inside a source text; OMEX uses spans everywhere for provenance.
- **Intent graph / AMT** — a tree of goals and requirements built by traversing grammar graphs (a host-system concept; see docs/INTEGRATIONS.md).
- **ANN / HNSW** — approximate nearest-neighbor search over embedding vectors; how a span is matched to candidate knowledge entries in sub-millisecond time.
- **Adapter / LoRA** — small add-on weight matrices that specialize a frozen model for a domain's *syntax* without retraining it (never for facts — see §7).
- **Knowledge Context Pool** — the bounded, scored subgraph of the knowledge fabric selected for one intent branch by grammar-anchored traversal (docs/KNOWLEDGE.md §3). This — not raw addresses — is what "knowledge aggregated on the branch" means.
- **Response Graph** — the handoff contract between reasoning and rendering: a grammar-schema skeleton whose content leaves are pre-filled with verified material carrying provenance (docs/RESPONSE_GRAPH.md).
- **Rendering ladder** — the three-tier rendering stack (template renderer → validated LLM stand-in → trained Realizer) that makes responses available from day zero (docs/REALIZER.md §6).
- **Cycle consistency** — the training loss that parses the Realizer's output with the frozen parser and penalizes any divergence from the input Response Graph's content.
- **Parse-back** — the production audit form of cycle consistency: parse a rendered response and verify full content coverage and **zero additions**.

---

---

# FILE: docs/ARCHITECTURE.md

---

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

---

# FILE: docs/FORMAT.md

---

# The OMEX Format

Three things carry the name: the **model package format** (how an OMEX model ships), the **graph interchange format** (what OMEX models read and write), and the **Response Graph** (the reasoning→rendering handoff, a first-class interchange structure with its own full specification in docs/RESPONSE_GRAPH.md). The interchange graphs are the most fundamental — they are the system's currency.

## 1. Model Package: `model.omex/`

```
model.omex/
├── metadata.json                 # name, version, model_class
│                                 # (grammar_parser | knowledge_linker |
│                                 #  realizer | modality_structural),
│                                 # language(s), schema versions,
│                                 # training-corpus provenance
├── schema/
│   ├── grammar_schema.json       # full node taxonomy + properties (see §4)
│   ├── edge_schema.json          # relationship/edge type registry:
│   │                             # cross-sentence relations, discourse
│   │                             # relations, per-modality edge sets
│   └── knowledge_ref_schema.json # knowledge-reference contract (kinds, path roles)
├── graph.json                    # the model's computation graph (architecture)
├── weights/
│   ├── quantization.json
│   ├── layer_0.bin
│   └── ...
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── merges.txt
├── heads/                        # task heads sharing the encoder
│   ├── structure.json / structure.bin
│   ├── correction.json / correction.bin
│   ├── knowledge_span.json / knowledge_span.bin
│   └── boundary.json / boundary.bin
├── io/
│   ├── input_contract.json       # text | graph | response-graph | span+context
│   └── output_contract.json      # graph | text | references
└── extensions/
    └── adapters/                 # optional structural (syntax-only) adapters
```

**The defining property:** `schema/` versions are what make two OMEX models compatible. Composition is legal exactly when the producer's output contract matches the consumer's input contract at compatible schema versions. There are **no** hardware profiles, memory budgets, or embedded optimizers anywhere in the package — those were removed from the design deliberately. Per docs/ARCHITECTURE.md §2, adapters in `extensions/` are structural (formatting/notation) only — a package shipping "knowledge adapters" is out of specification.

**Optional pipeline manifest.** A host may ship a `pipeline.json` listing model stages and their graph contracts (parser → linker → realizer, etc.). It is a composition declaration only — no runners, no memory allocations, no device hints.

## 2. Graph Interchange: A Worked Example (Parser Output)

Input sentence (with a deliberate error):

> "The dog quickly chase the cat through the garden."

Parser output (abbreviated where children repeat the same shape):

```json
{
  "schema_version": "1.0",
  "sentence": {
    "order": 1,
    "original_sentence": "The dog quickly chase the cat through the garden.",
    "span_start": 0,
    "span_end": 49,
    "corrected_sentence": "The dog quickly chased the cat through the garden.",
    "corrections": [
      { "error_type": "tense",
        "original_fragment": "chase",
        "corrected_fragment": "chased",
        "position_start": 16, "position_end": 21 }
    ],
    "sentence_type": "declarative",
    "grammar_tree": {
      "node_type": "Sentence",
      "text": "The dog quickly chased the cat through the garden.",
      "position_start": 0, "position_end": 50,
      "properties": {},
      "children": [
        { "node_type": "MainClause", "text": "...", "position_start": 0, "position_end": 49,
          "children": [
            { "node_type": "Subject", "text": "The dog", "position_start": 0, "position_end": 7,
              "children": [
                { "node_type": "NounPhrase", "text": "The dog", "position_start": 0, "position_end": 7,
                  "children": [
                    { "node_type": "Determiner", "text": "The", "position_start": 0, "position_end": 3,
                      "properties": { "definiteness": "definite" }, "children": [] },
                    { "node_type": "Noun", "text": "dog", "position_start": 4, "position_end": 7,
                      "properties": { "number": "singular" }, "children": [] } ] } ] },
            { "node_type": "Predicate", "text": "quickly chased the cat through the garden",
              "position_start": 8, "position_end": 49,
              "children": [
                { "node_type": "Adverb", "text": "quickly", "position_start": 8, "position_end": 15,
                  "properties": { "subtype": "manner" }, "children": [] },
                { "node_type": "MainVerb", "text": "chased", "position_start": 16, "position_end": 22,
                  "properties": { "tense": "past", "voice": "active", "polarity": "affirmative" },
                  "children": [] },
                { "node_type": "DirectObject", "text": "the cat", "position_start": 23, "position_end": 30,
                  "children": [ /* NounPhrase → Determiner + Noun */ ] },
                { "node_type": "PrepositionalPhrase", "text": "through the garden",
                  "position_start": 31, "position_end": 49,
                  "children": [
                    { "node_type": "Preposition", "text": "through", "position_start": 31, "position_end": 38, "children": [] },
                    { "node_type": "ObjectOfPreposition", "text": "the garden",
                      "position_start": 39, "position_end": 49,
                      "children": [ /* NounPhrase */ ] } ] } ] } ] },
        { "node_type": "Period", "text": ".", "position_start": 49, "position_end": 50, "children": [] }
      ]
    },
    "modality_spans": [],
    "knowledge_refs": []
  }
}
```

A knowledge-bearing example:

> "The annealing process relieves internal stress."

adds, alongside the tree:

```json
"knowledge_refs": [
  { "span_start": 4,  "span_end": 13, "surface": "annealing",
    "knowledge_kind": "process",
    "topic_path": ["Manufacturing","Metallurgy","Heat Treatment","Annealing"],
    "confidence": 0.97 },
  { "span_start": 31, "span_end": 46, "surface": "internal stress",
    "knowledge_kind": "concept",
    "topic_path": ["Engineering","Materials Science","Mechanics","Internal Stress"],
    "confidence": 0.94 }
]
```

Corpus-level structures (see docs/TRAINING.md for the full schema): `sections[]`, `paragraphs[]`, `cross_sentence_relationships[]`, `coreference_chains[]`.

**Tree constraints — enforced by the decoder, checkable by any consumer:** root is always `Sentence`; every node has exactly one parent; every child's span lies within its parent's; sibling spans do not overlap; every surface token belongs to exactly one leaf path; positions index into the corrected sentence; everything attaches to its grammatical parent (a relative clause under the noun phrase it modifies; punctuation under the sentence or clause it terminates).

## 3. The Response Graph (Interchange, Output Side)

The Response Graph is the input contract of the Realizer and of every renderer on the rendering ladder. Full specification, assembly algorithm, invariants, and worked examples live in **docs/RESPONSE_GRAPH.md**; the format essentials:

- A grammar-schema **skeleton** whose content-bearing leaves are **pre-filled** with verified material (knowledge-pool facts, task results), each carrying `knowledge_ref` / provenance metadata.
- **Two granularities per sentence:** `"frame"` (Subject/Verb/Object-level skeleton; the renderer chooses internal phrase structure and ordering variation — fluency headroom) and `"tree"` (a fully specified grammar tree per §2's constraints — exactness for legal text, quoted procedures, code). Both are legal in the same response.
- **Response-level `voice`** properties (tone, formality, warmth, directness, humor) and **`discourse.relations`** between sentences.
- **Renderer permissions (invariant):** function words, inflection/agreement, connectives realizing the declared discourse relations, ordering variation within frames — nothing else. **Parse-back** must recover every content leaf and surface no ungrounded span.

## 4. The Grammar Schema (Full Taxonomy)

Node types, by family:

- **Roots & clauses:** Sentence, MainClause, SubordinateClause, RelativeClause, ComplementClause, AdverbialClause, ConditionalClause, ComparativeClause, CoordinateClause, ParentheticalClause, EllipticalClause, QuotedClause
- **Phrases:** Phrase, NounPhrase, VerbPhrase, AdjectivePhrase, AdverbPhrase, PrepositionalPhrase, InfinitivePhrase, ParticipialPhrase, GerundPhrase, AbsolutePhrase, AppositivePhrase, CoordinatePhrase, ParentheticalPhrase
- **Predicate structure:** Predicate, Verb, MainVerb, AuxiliaryVerb, ModalVerb, LinkingVerb, HelpingVerb, PhrasalVerb, Copula, PredicateComplement, PredicateNominative, PredicateAdjective
- **Arguments:** Subject, ImpliedSubject, ExpletiveSubject, DirectObject, IndirectObject, ObjectComplement, SubjectComplement, Complement
- **Nouns:** Noun, CommonNoun, ProperNoun, CollectiveNoun, MassNoun, CountNoun, ConcreteNoun, AbstractNoun, CompoundNoun
- **Pronouns:** Pronoun, PersonalPronoun, ReflexivePronoun, ReciprocalPronoun, RelativePronoun, DemonstrativePronoun, InterrogativePronoun, IndefinitePronoun, PossessivePronoun
- **Determiners:** Determiner, Article, Demonstrative, PossessiveDeterminer, Quantifier, Numeral, DistributiveDeterminer
- **Modifiers:** Modifier, Adjective, AttributiveAdjective, PredicativeAdjective, ComparativeAdjective, SuperlativeAdjective, Adverb, AdverbOfTime, AdverbOfPlace, AdverbOfManner, AdverbOfDegree, AdverbOfFrequency, SentenceAdverb
- **Prepositions:** Preposition, SimplePreposition, CompoundPreposition, PhrasalPreposition
- **Conjunctions:** Conjunction, CoordinatingConjunction, SubordinatingConjunction, CorrelativeConjunction
- **Dependents:** RelativePronounClause, RelativeModifier, AdjectivalModifier, AdverbialModifier, NominalModifier, DeterminerModifier, PossessiveModifier, NumericModifier, QuantifierModifier, Apposition, Vocative, Parenthetical
- **Prepositional components:** PrepositionalObject, ObjectOfPreposition, ComplementOfPreposition
- **Verbal components:** InfinitiveMarker, Infinitive, Gerund, PresentParticiple, PastParticiple
- **Clause functions:** ClauseSubject, ClausePredicate, ClauseObject, ClauseComplement, ClauseModifier
- **Negation:** Negation, NegativeMarker, NegativeDeterminer, NegativePronoun
- **Questions:** QuestionMarker, InterrogativeWord, TagQuestion, WhPhrase
- **Comparison:** Comparison, ComparativeMarker, SuperlativeMarker, EqualityMarker
- **Coordination:** Coordination, Coordinator, CoordinatedElement
- **Agreement:** AgreementMarker, CaseMarker, NumberMarker, GenderMarker, PersonMarker
- **Punctuation:** Punctuation, Comma, Period, Colon, Semicolon, Dash, Hyphen, Parenthesis, Quotation, Apostrophe, Ellipsis, Exclamation, QuestionMark
- **Terminals:** Token, Word, Symbol, Number, Letter, Unknown

Properties (any node, all optional): `tense, aspect, mood, voice, person, number, gender, case, definiteness, comparison, polarity, subtype`.

**Relation registries in `edge_schema.json`:**

- **Cross-sentence relationship types:** `Elaborates, Causes, Enables, Prevents, Contradicts, Exemplifies, Summarizes, TemporalPrecedes, Coreference, PartOf, SimilarTo`.
- **Discourse relation types (Response Graph):** `Elaborates, Contrast, Cause, Sequence, Example, Summary` — the relations a renderer may realize as connectives ("however," "because," "then"); the *relations* are content, the connective *words* are surface realization.

## 5. Modality Edge Schemas (Illustrative)

The interchange principle extends per modality; `edge_schema.json` registers the modality's edge set. Examples:

- **Code:** Contains, Calls/CalledBy, Imports/ImportedBy, Defines/DefinedBy, DependsOn/RequiredBy, ImplementsPattern, UsesPattern, IntentSolves, DataFlows, ControlFlows, ExtendsAbstraction, plus cross-modal DescribedBy/Implements/ImplementedIn/VisualizedAs
- **Image:** Contains, PartOf/HasPart, SpatialAbove/Below/Left/Right, Occludes, PhysicallySupports, ContainedIn, FocalSubject, LeadsEyeTo, SimilarTo, cross-modal set — plus affordances ("CanSupport", "CanContain")
- **Audio:** Contains, Precedes/Follows, Causes, OverlapsWith, InterruptedBy, ResponseTo, ToneIndicates, PitchCorrelates, SyncedTo/SyncedBy — plus prosody analysis (pitch, rate, energy, emotional markers)
- **Video:** Contains, Precedes/Follows, InteractsWith, CollidesWith, FollowsSpatial, CausesEvent, NarrativeLeadsTo, AudioSyncsTo — plus object interactions (Touches, Collides, Follows, Approaches, Leaves, PassesBy) with time ranges
- **Math:** Contains, Precedes/Follows, DependsOn, LogicallyImplies, LogicallyEquivalent, Contradicts, Generalizes, SpecialCaseOf, UsedToProve, DischargesAssumption, AssumesIn

These are illustrative, not exhaustive — the registry is versioned and extensible per modality (full treatment: docs/MODALITIES.md).

---

---

# FILE: docs/TRAINING.md

---

# OMEX Training

## 1. Where the Graphs Come From

Three complementary sources:

**A. A graph-generation pipeline over real text.** An existing SLM/LLM is driven step-by-step over chunked text — identifying sentences one at a time in order, correcting each, detecting section markers and embedded non-prose spans, extracting each sentence's grammar tree, then pairwise cross-sentence relationships and coreference — with **every extracted structure validated by repeated independent yes/no confirmation before acceptance** (the reference implementation requires five consecutive confirmations per item). Only stable, validated structures enter the corpus; the validation loop *is* the data-QA gate. Pipeline output is additionally **calibrated against gold-standard human-annotated treebanks**: if factory agreement on the calibration set drops below the gate (≈95% — see docs/RISKS.md R2), corpus generation pauses until prompts or factory models are fixed. (The reference implementation of this pipeline lives in the Ozone-Studio host system — see docs/INTEGRATIONS.md — but any pipeline producing schema-conformant, validated graphs qualifies.)

**B. Synthetic generation at controlled coverage.** The master prompt below produces complete training examples per topic, register, and error density, scalable to hundreds of thousands of examples.

**C. The operational flywheel.** Once a system is live on the rendering ladder (docs/REALIZER.md §6), every validated Tier-0/Tier-1 rendering — a Response Graph paired with its parse-back-verified text — is a new `(graph → text)` training pair. **Operating the system manufactures the Realizer's corpus.** Deployments must gate flywheel collection behind explicit privacy policy (docs/DEPLOYMENT.md §6), and corpus provenance metadata must mark origin.

**Every example trains both directions.** Text→graph trains the parser; the same pair reversed trains the realizer. One corpus, two models — with the realizer-side asymmetry noted in §5.

## 2. Master Training-Data Generation Prompt

Self-contained; assumes no knowledge of OMEX; scaled by iterating `TOPIC × REGISTER × ERROR_INJECTION × seed`, with the tooling rules in §3 rejecting and regenerating invalid outputs.

```text
You are generating one complete training example for a grammar-graph language
system. Given a TOPIC, you will produce: (1) a source passage, (2) an ordered
sentence decomposition with grammar corrections, (3) one grammar tree per
sentence, (4) cross-sentence relationships and coreference chains, (5) knowledge
reference annotations, (6) structural annotations (sections and paragraphs), and
(7) candidate semantic units. Follow the schemas EXACTLY so automated tooling
can parse the output. Return ONLY one valid JSON object. No markdown. No
commentary.

INPUT PARAMETERS
TOPIC: {topic}
LANGUAGE: {language}
REGISTER: {conversational | technical | narrative | instructional | mixed}
ERROR_INJECTION: {none | light | moderate | heavy}
TARGET_SENTENCES: {integer, 8–40}

STEP 1 — SOURCE TEXT (internal, then emitted in the JSON)
Write a passage about TOPIC in REGISTER, in LANGUAGE, containing approximately
TARGET_SENTENCES sentences across 2–6 paragraphs. If REGISTER is technical or
instructional, include 1–3 section headings using any consistent structural
marker style (markdown, numbered, ALL-CAPS, underlined — vary across examples).
If ERROR_INJECTION is not "none", deliberately introduce realistic grammar and
spelling errors distributed across sentences at that density, drawn from varied
error types: subject-verb agreement, tense, article use, word order,
punctuation, run-ons, fragments, spelling. At least a third of sentences must
reference identifiable knowledge topics (named concepts, principles,
procedures, entities). Where natural for the topic, embed exactly one non-prose
span (a short code snippet or a mathematical expression) inside or between
sentences.

STEP 2 — OUTPUT SCHEMA
{
  "meta": {
    "schema_version": "1.0",
    "topic": "...", "language": "...", "register": "...",
    "error_injection": "...", "sentence_count": N
  },
  "source_text": "the full passage, exactly as written, errors included",
  "sections": [
    { "section_id": 1, "title": "..." , "level": 1,
      "marker_pattern": "description of the marker style",
      "start": <char offset in source_text>, "end": <char offset> }
  ],
  "paragraphs": [
    { "paragraph_id": 1, "section_id": 1 or null,
      "start": <char offset>, "end": <char offset> }
  ],
  "sentences": [
    {
      "order": 1,
      "paragraph_id": 1,
      "original_sentence": "exact text from source_text, character for character",
      "span_start": <char offset in source_text>,
      "span_end": <char offset in source_text>,
      "corrected_sentence": "corrected version; identical to original if no errors",
      "corrections": [
        { "error_type": "subject_verb_agreement|tense|article|spelling|
                         punctuation|word_order|fragment|run_on|other",
          "original_fragment": "...", "corrected_fragment": "...",
          "position_start": <offset in original_sentence>,
          "position_end": <offset in original_sentence> }
      ],
      "sentence_type": "declarative|interrogative|imperative|exclamatory|fragment",
      "grammar_tree": <GRAMMAR_NODE>,
      "modality_spans": [
        { "modality": "code|math|chemistry|...",
          "span_start": <offset in corrected_sentence>,
          "span_end": <offset>,
          "intent_reference": "contains|describes|references|mentions" }
      ],
      "knowledge_refs": [
        { "span_start": <offset in corrected_sentence>, "span_end": <offset>,
          "surface": "the span text",
          "knowledge_kind": "concept|principle|procedure|entity|process|tool|material",
          "topic_path": ["Domain","Discipline","Topic","Concept-or-leaf"],
          "confidence": 0.0-1.0 }
      ]
    }
  ],
  "cross_sentence_relationships": [
    { "from_order": i, "to_order": j,
      "relationship_type": "Elaborates|Causes|Enables|Prevents|Contradicts|
                            Exemplifies|Summarizes|TemporalPrecedes|Coreference|
                            PartOf|SimilarTo",
      "evidence": "brief quote or paraphrase demonstrating the relationship" }
  ],
  "coreference_chains": [
    { "chain_id": 1, "canonical_form": "...",
      "mentions": [
        { "sentence_order": i, "text": "...",
          "grammar_role": "subject|direct_object|indirect_object|modifier|
                           object_of_preposition",
          "span_start": <offset in that corrected_sentence>,
          "span_end": <offset> } ] }
  ],
  "amt_candidates": [
    { "candidate_id": 1,
      "unit_summary": "one-line description of the coherent semantic unit",
      "member_sentences": [orders...],
      "actor": "...", "action": "...", "target": "...",
      "constraints": ["..."],
      "supporting_relationships": [indices into cross_sentence_relationships] }
  ],
  "knowledge_graph_fragment": {
    "nodes": [ { "id": "n1", "kind": "Domain|Discipline|Topic|Concept|Principle|
                                       Procedure|Process|Tool|Material|Skill",
                 "label": "..." } ],
    "edges": [ { "from": "n1", "to": "n2",
                 "type": "contains|references|requires|produces|part_of|uses" } ]
  }
}

GRAMMAR_NODE (recursive; used for "grammar_tree"):
{
  "node_type": "<one of the full grammatical taxonomy: Sentence, MainClause,
    SubordinateClause, RelativeClause, ComplementClause, AdverbialClause,
    ConditionalClause, ComparativeClause, CoordinateClause, ParentheticalClause,
    EllipticalClause, QuotedClause, Phrase, NounPhrase, VerbPhrase,
    AdjectivePhrase, AdverbPhrase, PrepositionalPhrase, InfinitivePhrase,
    ParticipialPhrase, GerundPhrase, AbsolutePhrase, AppositivePhrase,
    CoordinatePhrase, ParentheticalPhrase, Predicate, Verb, MainVerb,
    AuxiliaryVerb, ModalVerb, LinkingVerb, HelpingVerb, PhrasalVerb, Copula,
    PredicateComplement, PredicateNominative, PredicateAdjective, Subject,
    ImpliedSubject, ExpletiveSubject, DirectObject, IndirectObject,
    ObjectComplement, SubjectComplement, Complement, Noun, CommonNoun,
    ProperNoun, CollectiveNoun, MassNoun, CountNoun, ConcreteNoun, AbstractNoun,
    CompoundNoun, Pronoun, PersonalPronoun, ReflexivePronoun, ReciprocalPronoun,
    RelativePronoun, DemonstrativePronoun, InterrogativePronoun,
    IndefinitePronoun, PossessivePronoun, Determiner, Article, Demonstrative,
    PossessiveDeterminer, Quantifier, Numeral, DistributiveDeterminer, Modifier,
    Adjective, AttributiveAdjective, PredicativeAdjective, ComparativeAdjective,
    SuperlativeAdjective, Adverb, AdverbOfTime, AdverbOfPlace, AdverbOfManner,
    AdverbOfDegree, AdverbOfFrequency, SentenceAdverb, Preposition,
    SimplePreposition, CompoundPreposition, PhrasalPreposition, Conjunction,
    CoordinatingConjunction, SubordinatingConjunction, CorrelativeConjunction,
    RelativePronounClause, RelativeModifier, AdjectivalModifier,
    AdverbialModifier, NominalModifier, DeterminerModifier, PossessiveModifier,
    NumericModifier, QuantifierModifier, Apposition, Vocative, Parenthetical,
    PrepositionalObject, ObjectOfPreposition, ComplementOfPreposition,
    InfinitiveMarker, Infinitive, Gerund, PresentParticiple, PastParticiple,
    ClauseSubject, ClausePredicate, ClauseObject, ClauseComplement,
    ClauseModifier, Negation, NegativeMarker, NegativeDeterminer,
    NegativePronoun, QuestionMarker, InterrogativeWord, TagQuestion, WhPhrase,
    Comparison, ComparativeMarker, SuperlativeMarker, EqualityMarker,
    Coordination, Coordinator, CoordinatedElement, AgreementMarker, CaseMarker,
    NumberMarker, GenderMarker, PersonMarker, Punctuation, Comma, Period, Colon,
    Semicolon, Dash, Hyphen, Parenthesis, Quotation, Apostrophe, Ellipsis,
    Exclamation, QuestionMark, Token, Word, Symbol, Number, Letter, Unknown>",
  "text": "the exact surface span this node covers",
  "position_start": <char offset within corrected_sentence>,
  "position_end": <char offset within corrected_sentence>,
  "properties": {
    "tense": null|"past|present|future",
    "aspect": null|"simple|progressive|perfect|perfect_progressive",
    "mood": null|"indicative|imperative|subjunctive|interrogative",
    "voice": null|"active|passive",
    "person": null|"first|second|third",
    "number": null|"singular|plural",
    "gender": null|"masculine|feminine|neuter|unknown",
    "case": null|"subjective|objective|possessive",
    "definiteness": null|"definite|indefinite",
    "comparison": null|"positive|comparative|superlative",
    "polarity": null|"affirmative|negative",
    "subtype": null|"free text refinement"
  },
  "children": [ <GRAMMAR_NODE>, ... ]
}

TREE CONSTRAINTS (mandatory):
- The root node of every grammar_tree is node_type "Sentence".
- Every node has exactly one parent (appears in exactly one children array).
- Every child's [position_start, position_end] lies within its parent's span.
- Sibling spans do not overlap.
- Nothing is duplicated: each surface token belongs to exactly one leaf path.
- Positions index into corrected_sentence.
- Attach every element to its grammatical parent (a relative clause attaches
  under the noun phrase it modifies; a prepositional phrase under what it
  modifies; punctuation under the sentence or the clause it terminates).
```

## 3. Tooling Validation Rules (reject and regenerate on violation)

- All spans index correctly into their reference string; sentence spans are ordered, non-overlapping, and jointly cover all non-heading prose of `source_text`.
- `corrections` is empty iff `original_sentence == corrected_sentence`.
- Grammar trees satisfy the tree constraints mechanically (single parent, containment, no sibling overlap, full leaf coverage).
- Every `knowledge_refs.topic_path` uses ontology role names consistently; every path node appears in `knowledge_graph_fragment`.
- Cross-sentence relationship endpoints exist; coreference mention spans resolve.

## 4. Scale and Curriculum (Parser)

Hundreds of thousands of examples = |topics| × |registers| × |error densities| × seeds. Curriculum order: clean single-clause sentences → multi-clause with subordination → error-injected → mixed-register documents with sections → knowledge-dense technical passages. Real pipeline output (validated graphs from actual prompts and documents) is interleaved with synthetic data so the parser sees natural distributions, not only generated text.

## 5. Realizer Training

The Realizer trains on the same corpus, reversed — with four rules that make it work (full architecture and risk treatment: docs/REALIZER.md):

1. **Corrected-side only.** The parser sees originals and corrections; the Realizer trains **only** on `(graph → corrected_sentence)`. It must never learn to emit errors.
2. **Voice variants, content-leaf-identical.** For voice conditioning, the same Response Graph is rendered at multiple voice settings (via the generation prompt with a voice parameter); variant construction guarantees the *content leaves are identical* across variants — only register differs — so the model mechanically cannot learn to express "warmth" by adding claims.
3. **Cycle-consistency loss.** Every training output is re-parsed by the frozen GrammarParser; divergence from the input graph's content set is penalized — missing leaves (coverage failure) and, with the heaviest penalty, **content spans not present in the graph** (additions). The parser is the Realizer's permanent adversarial auditor, in training and in production audits alike.
4. **The flywheel feeds it.** Source C from §1 — validated Tier-0/Tier-1 renderings from live systems — continuously extends the Realizer corpus with real-distribution Response Graphs.

**Realizer curriculum:** structure-only single frames → property-rich frames (tense/aspect/polarity/number) → tree-granularity exact rendering → multi-sentence discourse with relations and connectives → voice conditioning → modality content embedding (code/math leaves rendered verbatim within prose) → long mixed responses.

---

---

# FILE: docs/EXECUTION.md

---

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

---

---

# FILE: docs/PERFORMANCE.md

---

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

---

---

# FILE: docs/INTEGRATIONS.md

---

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

---

---

# FILE: docs/FAQ.md

---

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

###
