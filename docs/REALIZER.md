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
