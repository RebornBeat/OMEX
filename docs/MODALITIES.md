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

Modality graphs reference one another via the cross-modal edge set: `DescribedBy/Describes, ImplementedIn/Implements, VisualizedAs/Visualizes, SyncedTo/SyncedBy, ReferencedBy/References`. When a task spans text and code, each model emits its graph and the host stitches them: a text sentence node `—Describes→` a code function node. **Execution contexts expand through graph stitching; parameters never merge** (see ARCHITECTURE.md §8). Response Graphs may carry modality content leaves (a code block inside a prose answer), rendered verbatim under structural-adapter formatting.

## 6. Schema Registry and Roadmap

Every modality's node/edge sets live in the versioned `schema/edge_schema.json` registry of its models; composition legality is schema-version compatibility, identical to text. Build order: **text → code → math** (closed systems, corpus-manufacturable by the same factory pattern) → **relational models for image/audio/video** (dependent on chosen perception front-ends and entity-graph corpora).
