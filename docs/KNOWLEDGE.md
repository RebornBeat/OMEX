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
