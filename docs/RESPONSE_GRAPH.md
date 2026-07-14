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
