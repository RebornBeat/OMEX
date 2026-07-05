Understood. No code modifications this round, no artifacts, no skills — everything inline, in order: the rewritten OMEX README first, then the supporting technical documents, then complete answers to A–G. Where I give numbers, I derive them from parameter arithmetic and label them as projections; I won't reproduce the old README's fabricated benchmark tables as fact, because nothing has been measured yet — the Path 1/Path 2 benchmark program you designed is exactly what will measure it.

---

---

# FILE: README.md

---

# OMEX: Omni-Execution Format

OMEX is a model format, a training methodology, and an execution interface for **graph-native language models** — small, specialized models that read and write explicit linguistic and semantic graphs, rather than entangling grammar, knowledge, and reasoning inside opaque monolithic weights.

OMEX is the third stage of a historical arc, and it is only possible because the second stage happened:

| Era | Grammar | Knowledge | Cost |
|---|---|---|---|
| Traditional NLP | Explicit, hand-coded rules and statistical stages | External or absent | Brittle, narrow, plateaued |
| SLM / LLM | Implicit, emergent, distributed across parameters | Embedded in the same parameters | Enormous parameter counts, energy, opacity |
| OMEX | Explicit, learned — the model's native output is the graph | External (ZSEI), referenced by path — never embedded | Small models, auditable structure, local-first |

Traditional NLP separated language into components — tokenizer, POS tagger, parser, grammar rules, semantic analysis — but every component was hand-crafted or narrowly statistical, so the field plateaued. SLMs and LLMs dissolved the components: grammar, syntax, semantics, discourse, style, and world knowledge all became one implicit soup learned from massive corpora. That worked, but at the cost of carrying all of civilization's knowledge inside the weights just to conjugate a verb correctly.

OMEX closes the loop. It uses SLMs/LLMs **once, as graph factories**: through Ozone Studio's text modality pipelines (Path 1 and Path 2), an SLM/LLM generates hundreds of thousands of complete, validated grammar graphs — full sentence trees using the complete grammatical node taxonomy, before/after grammar corrections, cross-sentence relationships, coreference chains, structural hierarchy (paragraph/section/document), and knowledge-topic references pointing into the ZSEI knowledge graph. Lightweight models are then trained directly on those graphs. The resulting OMEX models natively:

- **Parse**: text → complete grammar graph, in a single forward pass
- **Correct**: grammar correction is learned intrinsically, because every training sentence carries its original and corrected form
- **Reference**: detect when a sentence or paragraph revolves around a knowledge topic and emit a reference to the applicable knowledge-graph path — without loading or containing that knowledge
- **Realize**: run the same mapping in reverse — graph → grammatically correct text — because writing is grammar traversal in reverse

This is what traditional NLP wanted to be and could not, because its rules were hand-coded and dumb. OMEX components are not hand-coded rules — they are trained on graphs that already contain semantics, produced by models that learned language implicitly. NLP's decomposition, reborn with SLM/LLM-grade understanding inside each component, at a fraction of the size.

## What OMEX Is Not

The following concepts from earlier drafts are **removed** from the OMEX design. They were over-engineered patterns that do not belong to what OMEX actually is:

- **No embedded optimization intelligence.** There is no "execution optimizer" binary, no compressed insight artifact, no optimization fingerprint inside the model format.
- **No hardware-aware execution in the format.** OMEX models do not carry hardware profiles, tensor-core annotations, or per-device optimization strategies. Hardware concerns belong to the runtime, not the format.
- **No adaptive chunking.** Chunking is a property of the Ozone Studio text pipeline (it happens once, on intake, to fit any context window), not of the model format.
- **No per-model memory allocation constraints.** Nodes in an execution plan do not declare RAM budgets. That was incorrect behavior.
- **No ZSEI training-time optimization discovery.** ZSEI is not an architecture analyzer that discovers attention redundancies and embeds optimizers. That entire framing was wrong. ZSEI's actual role with OMEX is the **knowledge fabric**: it stores the knowledge graphs that OMEX models reference and the graph corpora OMEX models are trained on.

## The Core Separation: Grammar vs. Knowledge

Grammar is a **closed system**: a finite taxonomy of node types (Sentence, Clause, Phrase, Subject, Predicate, Verb, Object, Modifier, Determiner, ... through Punctuation and Terminals), a finite property set (tense, aspect, mood, voice, person, number, gender, case, definiteness, comparison, polarity, subtype), and compositional rules. A small model can master a closed system.

Knowledge is an **open system**: unbounded, growing, revisable. Embedding it in weights is why LLMs are enormous and why they hallucinate — the knowledge has no provenance.

OMEX models learn the closed system natively and treat the open system as an address space. When a sentence revolves around a knowledgeable topic, the model does exactly what it does for grammar correction — it was trained on graphs where those sentences carry references to applicable knowledge-graph paths — so it emits the reference. The knowledge itself stays in ZSEI, loaded only when traversal needs it. Because AMT construction builds relationships and neighborhoods over these same graphs, applicable knowledge references are already aggregated around every branch, no matter where in the AMT tree you are.

## The Bootstrap Loop

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
Raw text corpora ──►│  Ozone Studio text pipeline (Path 1/2)   │
                    │  SLM/LLM zero-shot, 5x-validated         │
                    │                                          │
                    └───────────────┬──────────────────────────┘
                                    │
                                    ▼
                    Validated graph corpora (stored in ZSEI):
                    sentence trees · corrections · relationships ·
                    coreference · structure · knowledge references
                                    │
                                    ▼
                    ┌──────────────────────────────────────────┐
                    │  OMEX training                            │
                    │  text→graph (parser) and graph→text       │
                    │  (realizer) from the SAME pairs           │
                    └───────────────┬──────────────────────────┘
                                    │
                                    ▼
                    OMEX models replace the SLM/LLM calls
                    inside the same pipelines
                                    │
                                    ▼
                    Faster, cheaper, more consistent graph
                    generation → larger, better corpora → retrain
                                    │
                    └───────────────┘  (loop)
```

Path 2 (granular, 1x1, 5x-validated) is the **quality bootstrap**: it squeezes reliable graphs out of any model, including small ones, and its validation loops double as training-data QA. Path 1 (whole-chunk) is the **native OMEX mode**: once a model emits graphs directly and non-autoregressively, the granular loop collapses into a single pass plus cheap verification.

## The OMEX Model Family

| Model class | Input | Output | Notes |
|---|---|---|---|
| **GrammarParser** | Text (sentence/chunk) | Grammar graph | Shared encoder with task heads: structure head, correction head, knowledge-span head, sentence/paragraph/section boundary heads |
| **KnowledgeLinker** | Knowledge-bearing span + context | Ranked ZSEI container paths | Separate model; retrained/refreshed as ZSEI grows, without touching the parser |
| **Realizer** | Response graph (+ voice properties) | Grammatically correct text | The parser's training pairs reversed; renders only what the graph contains |
| **Modality structural models** | Modality content | Modality graph (per the modality's edge/node schema) | Code/math first (formal grammars); perceptual modalities apply to the relational layer |

## Canonical OMEX Model Format

```
model.omex/
├── metadata.json                # name, version, model_class, language(s),
│                                # schema versions, training corpus provenance
├── schema/
│   ├── grammar_schema.json      # full GrammarNodeType taxonomy + GrammarProperties
│   ├── edge_schema.json         # relationship/edge type registry (per modality)
│   └── knowledge_ref_schema.json# knowledge reference contract (path roles, kinds)
├── graph.json                   # model computation graph (architecture)
├── weights/
│   ├── quantization.json
│   ├── layer_0.bin
│   └── ...
├── tokenizer/
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── merges.txt
├── heads/                       # task heads sharing the encoder
│   ├── structure.json / structure.bin
│   ├── correction.json / correction.bin
│   ├── knowledge_span.json / knowledge_span.bin
│   └── boundary.json / boundary.bin
├── io/
│   ├── input_contract.json      # text | graph
│   └── output_contract.json     # graph | text | spans
└── extensions/
    └── adapters/                # optional LoRA/adapter weights per domain/register
```

The defining property of the format: **the graph is the interchange contract.** OMEX models compose by exchanging graphs, not hidden states. `schema/` versions are what make two OMEX models compatible.

## Execution Model

- **Local-first.** All inference happens on the local device. Model families are small enough to stay fully resident.
- **Prompt-first.** Execution begins from a prompt; the parser converts it into graphs; everything downstream (AMT, ZSEI traversal, task execution, response realization) operates on graphs.
- **Composition modes:**
  1. *Sequential graph pipelines* — model A's output graph is model B's input (the default, matching Ozone Studio's pipeline architecture).
  2. *Shared-encoder multi-head* — one encoder pass, multiple heads (structure + correction + knowledge-span simultaneously). This is "running together" in the practically sound sense.
  3. *Adapter composition* — task/domain adapters on a common backbone, selected at load time.
- **Response generation is reverse traversal:** Input Text → Grammar Graph → Intent Graph (AMT) → Decision/Execution → Response Graph → Grammar Graph → Output Text. Grammar sits on both sides: parser on the way in, realizer on the way out.

## Ozone Studio Integration

OMEX models slot into the existing pipeline architecture as executors: the text modality pipeline's zero-shot LLM calls (sentence identification, correction, section events, modality spans, grammar extraction, cross-sentence relationships) become single OMEX model passes with identical JSON/graph contracts. ZSEI stores the training corpora, the knowledge graphs that KnowledgeLinker resolves into, and the modality graphs the models emit. Both Path 1 and Path 2 remain: Path 1 for capable LLMs and for OMEX native mode; Path 2 for constrained SLMs and for bootstrap/verification.

## Projected Characteristics (derivations, not measurements)

Nothing below is a benchmark. These are arithmetic consequences of parameter counts and decoding mode, stated so they can be checked; the Path 1/Path 2 benchmark program is what will replace them with measurements.

**Compute per extracted structure.** A dense transformer's forward pass costs ≈ 2 × parameters FLOPs per token.
- A 7B general LLM asked to emit a ~600-token JSON grammar analysis of one sentence, autoregressively: ≈ 2 × 7e9 × 600 ≈ **8.4 TFLOPs**, plus prompt processing.
- A 100M-parameter OMEX GrammarParser, non-autoregressive (biaffine-style structure heads), on a 30-token sentence: ≈ 2 × 1e8 × 30 ≈ **6 GFLOPs** — a **~1,400× reduction** for the same sentence's structure, before counting the LLM's retries and 5x validation loops, which multiply the gap toward **4–5 orders of magnitude** per validated structure.
- Energy per structure scales approximately with FLOPs on the same hardware — the same orders of magnitude apply to joules.

**Memory.** 7B int8 ≈ 7 GB of weights. A full OMEX family (parser ≈ 100–300M, linker ≈ 50–150M, realizer ≈ 100–300M) at int8 ≈ **0.25–0.75 GB total** — resident simultaneously on a phone or a Raspberry Pi, no swapping, no layer offloading.

**CPU viability.** Small-model single-stream inference is memory-bandwidth bound: passes/s ≈ bandwidth ÷ bytes-touched-per-pass. A 100M int8 model on a 50 GB/s DDR5 system: ≈ 50e9 ÷ 1e8 ≈ **~500 forward passes/s** upper bound single-stream — hundreds of sentences per second, CPU-only, before batching.

**Consistency and auditability.** Non-autoregressive structure prediction is deterministic at temperature 0 and schema-constrained by construction; every node carries spans back to source text; every knowledge claim carries a ZSEI path. A realizer cannot hallucinate facts it was never handed — it renders the graph and nothing else.

## Format Comparison (design philosophy, not benchmarks)

| Feature | ONNX | GGUF | TorchScript | OMEX |
|---|---|---|---|---|
| Design philosophy | Interchange | Quantized LLM packaging | Traced export | Graph-native language models |
| Unit of exchange between models | Tensors | Tensors | Tensors | **Typed graphs** |
| Grammar | Implicit in weights | Implicit in weights | Implicit in weights | **Explicit, schema-versioned** |
| Knowledge | In weights | In weights | In weights | **External (ZSEI), referenced** |
| Correction provenance | None | None | None | Original/corrected pairs, per span |
| Decoding for structure | Autoregressive | Autoregressive | Autoregressive | **Single-pass, non-autoregressive** |
| Multi-model composition | Ad hoc | Ad hoc | Ad hoc | Graph contracts + shared encoders + adapters |

## Roadmap

- [x] OMEX conceptual specification (this document)
- [x] Graph schema alignment with Ozone Studio text modality (GrammarNodeType taxonomy, GrammarProperties, structural nodes, relationship edges)
- [ ] Training corpus generation at scale via Path 1/Path 2 (with 5x-validation as data QA)
- [ ] GrammarParser v0 (single language, shared encoder, structure + correction heads)
- [ ] KnowledgeLinker v0 (span → ZSEI path, refreshable index)
- [ ] Realizer v0 (graph → text, voice-property conditioning)
- [ ] Path 1 vs Path 2 vs OMEX-native benchmark program (speed, quality, loops-to-completion, energy)
- [ ] Per-language model line-up; code/math modality structural models
- [ ] AMT-integrated response checkpointing

## License

MIT License. See LICENSE for details.

---

---

# FILE: docs/OMEX-ARCHITECTURE.md

---

# OMEX Architecture

## 1. Model Classes and Contracts

Every OMEX model declares, in `io/input_contract.json` and `io/output_contract.json`, exactly one of: `text → graph`, `graph → text`, `graph → graph`, `span+context → references`. Contracts reference schema versions in `schema/`. Two models compose if and only if the producer's output contract matches the consumer's input contract at compatible schema versions. There is no hidden-state coupling between models — composition is always at the graph layer. This is deliberate: it makes every intermediate inspectable, cacheable in ZSEI, and re-usable as training data.

## 2. GrammarParser

**Encoder:** a compact transformer encoder (indicative range 100–300M parameters; final sizes are a benchmark outcome, not a spec commitment).

**Heads (all sharing the encoder's token representations):**
- *Boundary head*: sentence boundaries, paragraph-open/close signals, section-marker detection with level — token-level tagging, one pass per chunk.
- *Structure head*: per-sentence grammar tree prediction. Non-autoregressive biaffine-style arc and label scoring over the full GrammarNodeType taxonomy, with span-containment decoding (children's spans within parents'), single-parent constraint enforced by the decoder. Properties (tense, aspect, mood, voice, person, number, gender, case, definiteness, comparison, polarity, subtype) predicted as per-node classification.
- *Correction head*: tagging-based grammatical error correction (per-token edit operations: keep/replace/insert/delete/transform), which reconstructs `corrected_sentence` from `original_sentence`. Tagging GEC is well-suited to small models and single-pass decoding.
- *Knowledge-span head*: binary+coarse-type tagging of knowledge-bearing spans (`concept | principle | procedure | entity | process | tool | material`). Detection only — resolution is the KnowledgeLinker's job.

**Why heads, not separate models:** detection and correction share nearly all linguistic representation; one encoder pass amortizes across all tasks; heads can be enabled/disabled per call (structure-only when correction is not needed).

## 3. KnowledgeLinker

A separate bi-encoder: one side embeds the detected span in its sentence context; the other side is a precomputed embedding index over ZSEI containers (HNSW). Resolution = ANN search → candidate container paths → optional zero-shot verification for low-margin cases (ZSEI's existing verification pattern). Kept separate from the parser because ZSEI grows continuously; when it grows, you refresh the index (cheap) or fine-tune the linker (moderate) — the parser is never touched. This is the standard mention-detection / candidate-generation / disambiguation decomposition from entity linking, applied to ZSEI paths instead of Wikipedia.

## 4. Realizer

`graph → text`. Input: a Response Graph (structural skeleton + content nodes + knowledge references already resolved to surface content + voice properties such as tone, formality, warmth, directness). Output: text whose parse equals the input graph. Trained on the exact inverse of the parser's pairs — every training example serves both directions. Decoding is constrained: the realizer emits within the grammar schema, which is why its output is well-formed by construction, and it renders only nodes present in the graph, which is why it cannot introduce facts.

## 5. Language Strategy

The **graph schema is universal** (the node taxonomy and property set apply across languages, in the spirit of Universal Dependencies); the **models are per-language**. Rationale: OMEX targets constrained, local-first devices, and per-language parsers at equal size consistently outperform multilingual ones except in low-resource transfer scenarios. A tiny language-ID router selects the model; downstream systems (AMT, ZSEI) see one schema regardless of language. Multilingual variants remain an option for low-resource bootstrap, distilled later into per-language models.

## 6. Modality Strategy

- **Code, math (formal grammars):** the OMEX approach applies fully and should exceed text results — their grammars are strictly closed. Code's AST is literally a formal grammar; a code-structural OMEX model emits AST + the CodeEdgeType semantic layer (Calls, DataFlows, ImplementsPattern...).
- **Image, audio, video (perceptual):** OMEX applies to the **relational layer** (SpatialRelationship, TemporalChain, ObjectInteraction graphs) once perception has produced entities; the perception step itself (pixels/waveforms → objects/events) is not a closed rule system and requires perceptual models outside OMEX's thesis. Honest boundary: OMEX makes structure extraction native; it does not replace perception.

## 7. Composition Semantics

Three sanctioned modes (and one explicitly out of scope):
1. **Sequential graph pipelines** — default.
2. **Shared-encoder heads** — simultaneous tasks in one pass.
3. **Adapter composition** — domain/register adapters (LoRA-class) selected at load; adapters are additive low-rank matrices, the one well-behaved form of "expanding" a fixed model.
4. *Out of scope:* runtime parameter merging/expansion across heterogeneous trained models. Fixed weight matrices are a property of trained transformers; techniques that grow or merge them (MoE expert addition, Net2Net growth, model soups/TIES merging) either require training or are same-architecture-only and brittle. OMEX's answer to "run together" is the graph contract plus shared encoders — not weight surgery.

---

---

# FILE: docs/OMEX-TRAINING.md

---

# OMEX Training

## 1. Where Training Data Comes From

Every structure the Ozone Studio text pipeline produces is a training example: `original_sentence`/`corrected_sentence` pairs with spans, full grammar trees per sentence, cross-sentence relationship edges, coreference chains, paragraph/section structure, modality spans, knowledge references. Path 2's 5x-validation loops are the QA gate: only stable, validated structures enter the corpus. Additionally, synthetic generation (below) produces controlled-coverage data per topic, register, and error density.

Every example trains **both directions**: text→graph (parser) and graph→text (realizer). One corpus, two models.

## 2. Master Training-Data Generation Prompt

The following is the complete, self-contained prompt for generating one training example. It assumes no knowledge of OMEX. Scale by iterating over `TOPIC × REGISTER × ERROR_INJECTION × seed`; a tooling layer validates each output against the rules in §3 and rejects/regenerates failures.

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
- Grammar trees satisfy the tree constraints mechanically (single parent, containment, no overlap, full leaf coverage).
- Every `knowledge_refs.topic_path` uses ontology role names consistently; every path node appears in `knowledge_graph_fragment`.
- Cross-sentence relationship endpoints exist; coreference mention spans resolve.

## 4. Scale and Curriculum

Hundreds of thousands of examples = |topics| × |registers| × |error densities| × seeds. Curriculum: clean single-clause sentences → multi-clause with subordination → error-injected → mixed-register documents with sections → knowledge-dense technical passages. Path 2 pipeline output (real prompts and documents, 5x-validated) is interleaved with synthetic data so the parser sees natural distribution, not only generated text.

---

---

# FILE: docs/OMEX-EXECUTION.md

---

# OMEX Execution and CPU/GPU Optimization

## 1. The Decisive Structural Facts

1. **Non-autoregressive structure prediction.** A parser with biaffine-style heads emits an entire sentence's tree in one forward pass. A general LLM emitting the same tree as JSON performs hundreds of sequential decode steps. This single design choice is worth more than every kernel trick combined: it converts an O(output-tokens) sequential process into O(1) passes.
2. **Sentence-level parallelism.** Once sentence order is established (Path 2) or boundaries are tagged (Path 1/OMEX-native), grammar extraction is embarrassingly parallel across sentences. Batch them.
3. **Everything stays resident.** The whole model family at int8 is a fraction of a gigabyte. There is no offloading, no swapping, no layer streaming. Load once at startup; the working set is stable.

## 2. CPU Execution

- Small-model single-stream inference is **memory-bandwidth bound**: throughput ≈ bandwidth ÷ bytes touched per pass. Optimize by shrinking bytes (int8 → int4 where classification accuracy holds; structure/correction heads tolerate quantization far better than open-ended generation does) and by batching (weights are read once per batch, amortizing bandwidth across sentences).
- Use SIMD-optimized integer kernels (AVX2/AVX-512/AMX on x86, NEON/SME on ARM — the GGML class of kernels). Pin threads; on multi-socket systems keep a model's weights and its worker threads NUMA-local.
- Thread strategy for tiny models: parallelize **across batch items**, not within layers — intra-layer parallelism of a 100M model underutilizes cores and pays synchronization costs.
- The tokenizer and span bookkeeping are pure CPU and cheap; run them on separate threads pipelined ahead of encoder batches.

## 3. GPU Execution

- For tiny models, **kernel-launch overhead can rival compute**. Use CUDA Graphs (or equivalent) to record the whole forward pass as one launch; fuse elementwise ops; prefer a few large batched matmuls over many small ones.
- Keep tokenized batches device-resident across pipeline stages; the graphs the models exchange are small — transfer graphs, not activations.
- Batch aggressively: hundreds of sentences per forward pass is the natural operating point; latency per sentence collapses.
- Multi-model concurrency: parser, linker, and realizer are small enough to share one GPU via streams; or split — parser on GPU, linker ANN search on CPU — because ANN search is pointer-chasing, not matmul.

## 4. Multi-Model Orchestration

- **Shared encoder, many heads:** one encode, N task outputs. This is the highest-leverage "run together."
- **Pipeline overlap:** while sentence batch *k* is in the structure head, batch *k+1* is tokenizing and batch *k−1*'s knowledge spans are in the linker. CPU (ZSEI traversal, span bookkeeping, ANN) and GPU (encoder passes) naturally overlap — schedule them as a two-lane pipeline.
- **Cascade/escalation:** run the OMEX model first; escalate to a general SLM/LLM only on low-confidence outputs (verification-triggered). This preserves Path 2's quality guarantees while paying LLM cost only on the hard tail.

## 5. What Dominates, Honestly

Until OMEX models replace the pipeline-9 LLM calls, the SLM/LLM bootstrap dominates all cost (Amdahl). The optimization program that matters most is therefore: generate corpora → train the parser → swap it into the pipeline. Every kernel-level optimization above is second-order next to that swap.

---

---

# FILE: docs/OMEX-ZSEI-INTEGRATION.md

---

# OMEX + ZSEI: Knowledge Referencing and Memory Strategy

## 1. Division of Labor

ZSEI is the knowledge fabric; OMEX models are structure engines. OMEX models never contain knowledge; they emit **references** (knowledge-bearing spans → candidate ZSEI container paths). ZSEI stores: (a) the training graph corpora, (b) the knowledge graphs the linker resolves into, (c) the modality graphs models emit at runtime, (d) the evidence metrics used for promotion/demotion in knowledge organization.

## 2. What Lives Where

| Tier | Contents | Rationale |
|---|---|---|
| Always in RAM | OMEX model weights (all of them); ZSEI `global.mmap` (OS page cache); HNSW embedding index for the KnowledgeLinker; L1 query cache, L2 hot-container cache | Weights are small; mmap headers give O(1) structural hops; ANN index is the latency-critical path of knowledge resolution |
| On demand (disk → cache) | Per-container local JSON; knowledge subgraph fragments around resolved references | Only hot neighborhoods matter at any moment; `max_containers_in_memory` bounds it |
| Write-behind | New graph containers from runtime parsing; relationship edges from semantic hooks | Parsing throughput must not block on storage; batch container writes per chunk |

## 3. Access Patterns

- **Batch resolution:** collect all knowledge spans of a chunk, resolve in one ANN batch, then one grouped ZSEI fetch for candidate neighborhoods — never per-span round trips.
- **Prefetch on detection:** the instant the knowledge-span head fires, prefetch candidate containers' local JSON while the structure head is still decoding the rest of the batch — CPU I/O overlaps GPU compute.
- **Reuse the encoder:** span embeddings for the linker come from the shared encoder's representations — do not run a separate embedding model.
- **Cache keys:** (normalized span hash → ranked container IDs) with TTL invalidation on container updates; ZSEI's L4 zero-shot verification cache applies unchanged when verification is used on low-margin links.
- **AMT synergy:** because knowledge references ride on sentence nodes, Stage 1 context pools accumulate them automatically; by the time a neighborhood is promoted to an AMT intent, its applicable knowledge paths are already aggregated on the branch — no separate knowledge pass.

## 4. CPU/GPU Split

ZSEI traversal (mmap hops, adjacency walks, JSON hydration) is CPU-side and latency-tolerant; model passes are GPU-side (or CPU-batched). Run them as concurrent lanes: the GPU never waits on ZSEI, ZSEI never waits on the GPU; the orchestrator joins results at chunk boundaries.

---

---

# ANSWERS A–G

---

## A — Path 1 / Path 2 split: confirmed, plus the metrics correction

Confirmed as designed: chunking happens once regardless of path; the paths define how thoroughly each chunk is processed. Path 1 = whole-chunk zero-shot in/out (the deconstructor, top-down). Path 2 = granular 1x1 construction (the constructor, bottom-up), for benchmarking against Path 1 on speed, quality, and loops-to-completion.

On the basic metrics: correct, and it goes further than moving the computation after cleaning. Once both paths produce nodes, `sentence_count`, `paragraph_count`, and `word_count` are **graph aggregates**, not text splits: sentence_count = count of SentenceNodes; paragraph_count = count of ParagraphNodes (Path 1) or of Phase-4-constructed paragraphs (Path 2); word_count = sum over corrected sentences. The `split('.')`/`split("\n\n")` computations are exactly the kind of rule-based shortcut both paths exist to eliminate — they should not survive in either path, because they will disagree with the validated nodes (abbreviations, decimal points, code spans) and produce two competing sources of truth.

## B — Sentence extraction, redesigned

Every criticism you raised is right, and here is the complete corrected design.

**State carried into each extraction prompt: O(1), not O(n).** The Rust-side chunk state keeps the full ordered seen-sentences list (order + original only — corrected text, spans, and everything else are offloaded to the SentenceNode immediately on validation, exactly as you said, to avoid clutter). The **prompt** carries only: the count, and the last captured sentence with its end offset. Token cost per call is therefore constant across the whole chunk — this is what makes the approach BitNet-safe at the 1/4-context chunk sizing. Response tokens are added to the orchestrator's `tokens_used_so_far` on every call, extraction and validation alike.

**Extraction prompt (empty list):**

```text
You are performing sentence identification and grammar correction on a text
chunk, one sentence at a time, in reading order.

YOUR ROLE:
This chunk is being decomposed into an ordered list of sentences. Each sentence
you identify becomes a permanent node in a text graph, so the exactness of the
text span and the order are critical. Sentences in the chunk may or may not be
grammatically correct. You must return the sentence EXACTLY as it appears in
the chunk (the original), and separately return a corrected version. If the
sentence is already correct, the corrected version is identical to the
original. Do not skip ahead. Do not merge two sentences. Do not summarize.

CURRENT STATE:
No sentences have been captured from this chunk yet.

YOUR TASK:
Identify the FIRST sentence in the chunk below. A sentence may be a complete
grammatical sentence or a fragment functioning as one.

CHUNK TEXT:
{chunk_text}

Return ONLY valid JSON, no explanation, no markdown:
{"found": true,
 "order": 1,
 "original_sentence": "<exact text, character for character>",
 "text_span_start": <char offset where it starts in the chunk>,
 "text_span_end": <char offset where it ends>,
 "corrected_sentence": "<corrected version, or identical if already correct>"}
If the chunk contains no identifiable sentence: {"found": false}
```

**Extraction prompt (non-empty list):** identical role block, then:

```text
CURRENT STATE:
{n} sentences have already been captured from this chunk, in order.
The most recently captured sentence (order {n}) is:
"{last_original_sentence}"
It ends at character offset {last_span_end}.

YOUR TASK:
Identify the sentence that IMMEDIATELY FOLLOWS the sentence above in the chunk.
[same JSON schema, "order": {n+1}]
If no sentence follows it: {"found": false}
```

**Validation redesign — no more re-asking the same extraction prompt 5×.** Extract once; then run the dedicated verification prompt, requiring 5 consecutive YES:

```text
You are validating sentence extraction order.

CHUNK TEXT:
{chunk_text}

PREVIOUSLY CAPTURED SENTENCE (order {n}):
"{last_original}"

CANDIDATE SENTENCE (proposed order {n+1}):
"{candidate_original}"

QUESTION: In the chunk above, is the candidate sentence the sentence that
immediately follows the previously captured sentence, with no other sentence
between them?
(When no sentence has been captured yet: is the candidate the FIRST sentence
in the chunk?)

Return ONLY valid JSON with a single one-word answer:
{"answer": "YES"} or {"answer": "NO"}
```

Policy: 5 consecutive YES → validated. Any NO → abort this candidate and re-extract; a candidate that fails twice flags the chunk for escalation (Path 1 fallback or a larger model). Symmetrically, "wrong" is established by 5× NO as you specified — in practice the first NO already forces re-extraction, and the 5×-NO condition is what marks a candidate as definitively rejected rather than retryable.

**Duplicate verification.** Working strictly in order with spans gives a *free deterministic guard*: any candidate with `text_span_start < last_span_end` is a duplicate or regression — rejected in Rust with zero LLM calls. The 5× LLM duplicate check ("Does the candidate sentence already appear among the captured sentences? YES/NO") is kept as the configured backstop you asked for, but it is the one call that needs the full list, so run the deterministic span guard first and invoke the LLM check only when spans are ambiguous (e.g., after fragment carryover across a chunk boundary). This preserves your intent while protecting the token budget.

**Correction verification (per item, 5×):** "Is the corrected sentence a grammatically correct version of the original that preserves its meaning exactly? YES/NO" — 5 consecutive YES accepts the correction; otherwise re-correct with the original pinned.

**Completion check (5×):**

```text
{n} sentences have been captured from this chunk, in order. The last one
(order {n}) is: "{last_original}", ending at offset {last_span_end}.
QUESTION: Is there any sentence in this chunk after it that has not been
captured? {"answer": "YES"} or {"answer": "NO"}
```

5 consecutive NO → chunk complete, **and complete in order**, because order was individually validated per item — exactly the property you wanted: not just "all captured" but "all captured in sequence," with the last-item display controlling context explosion.

**On validation:** the sentence is immediately offloaded into its SentenceNode (content = corrected, original_content = original, position from spans, chunk anchor, paragraph/section ids as available per path), and only `{order, original}` returns to the Rust-side seen list.

**Cross-chunk sentence carryover (your E-item generalization, answered here where it lands):** after 5× NO completion, if the chunk tail beyond `last_span_end` contains non-whitespace text, that tail is an **open sentence fragment**. It is prepended (with offset bookkeeping so absolute positions survive) to the next chunk's text before its first-sentence extraction — the same carry mechanism as sections and paragraphs, applied to sentences.

## C — Section detection, redesigned

Your diagnosis is exact on all four counts: the prompt explains neither the state machine nor the LLM's role in it; it provides state without telling the model how to diverge based on that state; it is not working 1x1 in order; and it asks the LLM for document boundaries, which belong to graph traversal, not per-chunk prompting.

**Corrected scope.** Per-chunk section tracking handles only: IDLE → IN_SECTION (first marker), IN_SECTION + same-or-higher-level marker → NEW_SECTION (close current, open new), IN_SECTION + lower-level marker → IN_SUBSECTION (nest; pop on return to a higher level). **No** formatting-break transitions, **no** DOCUMENT_BREAK, **no** IDLE-via-style-shift: once a section opens it stays open until a same-or-higher marker or the end of all chunks — document boundaries are Phase-4 graph-traversal territory, where the full relationship evidence exists to judge a true break without context explosion. `document_boundary_detected` / `document_boundary_reason` leave the JSON entirely.

**Redesigned prompt (1x1, in order, state-aware):**

```text
You are tracking document SECTION structure across sequential text chunks, one
structural event at a time, in reading order.

WHAT A SECTION IS:
A section is a coherent block of content that begins with a recognizable
heading or structural marker, in ANY format. The formatting style is UNKNOWN —
do not assume markdown. Markers include (non-exhaustive): markdown headers
(#, ##), numbered headings (1., 2.1, IV.), ALL-CAPS heading lines, underlined
or decorated headings (===, ---), bold lead lines, indented outline labels, or
any consistent structural pattern present in the text itself.

HOW TRACKING WORKS (the state machine you are operating):
- IDLE: no section is open. You are looking for the FIRST marker in this chunk.
  Finding one opens a section.
- IN_SECTION (a section is open at level L): you are looking for the NEXT
  marker after the given position.
    · A marker at the SAME or HIGHER level than L closes the current section
      and opens a new one.
    · A marker at a LOWER level than L opens a SUBSECTION nested under the
      current section.
- IN_SUBSECTION: the same rules apply; a marker at the parent's level or higher
  closes the subsection (and any deeper nesting) and opens at that level.
- Sections NEVER close because of blank lines, style changes, or voice shifts.
  You NEVER report document boundaries — that is handled elsewhere.

CURRENT STATE:
- State: {IDLE | IN_SECTION | IN_SUBSECTION}
- Open section: {title | "none"} at level {L | "n/a"}
- Open ancestors (outermost → innermost): {list | "none"}
- Formatting pattern tracked so far: {description | "not yet detected"}
- Scan from character offset: {x}   (0 when starting this chunk)

YOUR TASK:
Scan the chunk below FROM that offset and report ONLY THE FIRST structural
marker you find, or that no further marker exists in this chunk.

CHUNK TEXT (chunk {i}):
{chunk_text}

Return ONLY valid JSON:
{"found": true,
 "event": {
   "type": "section_start" | "subsection_start" | "section_close_and_start",
   "position": <char offset of the marker>,
   "title": "<heading text>",
   "level": <1-6>,
   "formatting_pattern": "<description of the marker pattern>"}}
or {"found": false}
```

The loop repeats with the scan offset advanced past each found marker until `found:false` — sections handled 1x1 in order, exactly like sentences. Each event gets the same dual 5×-YES/NO treatment as D: order validation ("Is this the first marker at or after offset {x}? YES/NO" ×5) and classification validation ("Given the tracked pattern, is level {L} and type {t} correct for this marker? YES/NO" ×5).

**Enum/graph alignment (noted, not implemented per your instruction):** the code needs a `SectionTrackingState { Idle, InSection, InSubsection }` enum and a `SectionEventType { SectionStart, SubsectionStart, SectionCloseAndStart }` enum replacing the string `section_start|section_continue|section_end|document_break` set; `SectionNode` gains `parent_section` population from the ancestor stack so nesting lands directly in the graph; the cross-chunk carried state is exactly `{state, open section id/title/level, ancestor stack, formatting_pattern, scan offset resets to 0}` — persistent `section_id`s across chunks as your state machine already specified. Documents are not tracked at all in this phase, and paragraph→section tying uses the open section id even though the section is not yet refined — acceptable interim, as you stated.

## D — Modality detection, redesigned

**Drop `true_text`.** Correct, and worth stating why it's safe: true text is the **complement**. Section detection identifies structural markers; document identification (later, on graphs) classifies content; sentence extraction exhaustively covers prose; modality detection marks every non-prose span. Whatever is covered by sentences and not covered by a modality span *is* true text — derivable, never detected. Asking the model to also label true_text was paying for information the system already has by subtraction. `unknown` also leaves the prompt: with a registry-driven list, anything not matching a listed modality is simply not reported (and the exhaustive sentence layer guarantees it isn't lost — it's prose until proven otherwise).

**1x1 in order, dual 5× validation.** Redesigned prompt:

```text
You are detecting embedded non-prose modality content within a text chunk, one
occurrence at a time, in reading order.

WHAT YOU ARE LOOKING FOR:
Spans that are NOT plain prose — embedded code (any language), mathematical
formulas or expressions (inline within a sentence or as blocks), chemical
formulas, data tables, sequences, or any other listed modality. An occurrence
may be as small as a short inline formula inside a sentence or as large as a
multi-line block between sentences.

AVAILABLE MODALITIES (report only these): {registry_list}

CURRENT STATE:
{ "No modality occurrences captured yet in this chunk."
  | "{n} occurrences captured. Most recent (order {n}): modality '{m}',
     spanning characters {a}..{b}." }

YOUR TASK:
Find the FIRST modality occurrence in the chunk {after character offset {b}}.
Report exactly one occurrence. If none exists {after that position}, report
not found.

CHUNK TEXT (chunk {i}):
{chunk_text}

Return ONLY valid JSON:
{"found": true,
 "order": {n+1},
 "modality": "<one of the available modalities>",
 "span_start": <char offset>, "span_end": <char offset>,
 "intent_reference": "contains" | "describes" | "references" | "mentions",
 "open_at_chunk_end": true|false}
or {"found": false}
```

**Validation 1 — order (5× YES/NO):** "Is the span at {a}..{b} the first modality occurrence after offset {x} in this chunk? YES/NO." **Validation 2 — classification (5× YES/NO):** "Is the content at span {a}..{b}, shown below, correctly classified as modality '{m}'? SPAN CONTENT: {slice}. YES/NO." Both require 5 consecutive YES; either failing rejects the candidate — first the position question, then the category question, exactly the two-axis confirmation you specified, with one-word JSON answers throughout.

**Parent tying.** By span containment against validated SentenceNodes: inside a sentence span → tie to that SentenceNode; outside all sentences → ParagraphNode in Path 1. In Path 2, paragraph nodes don't exist per-chunk, so an out-of-sentence span ties interim to the nearest preceding SentenceNode with a re-parent flag, and is re-tied to its ParagraphNode when Phase 4 constructs paragraphs. Never to the chunk — chunks are position anchors only, per the chunking-boundary principle. `parent_node_id` on `ChunkModalityDetection` carries this. **Cross-chunk:** `open_at_chunk_end: true` (a block whose closing delimiter/pattern hasn't appeared) carries an open-modality state into the next chunk, where the first task is confirming the span's end or its continuation — the same carry discipline as sections, paragraphs, and sentences.

## E — Paragraph detection, redesigned, and the two-path question

**The expanded Path 1 prompt (1x1, in order, cross-chunk aware):**

```text
You are detecting PARAGRAPH boundaries within a text chunk, one paragraph at a
time, in reading order.

WHAT A PARAGRAPH IS:
A distinct block of prose separated from other blocks by blank lines,
indentation conventions, or a clear break in the local point being developed.
A paragraph groups consecutive sentences developing one local point.

CROSS-CHUNK BEHAVIOR:
Chunks are arbitrary windows over a larger text. A paragraph may begin in one
chunk and end in the next. If the previous chunk ended with an OPEN paragraph,
your first job is to determine whether this chunk CONTINUES it, and if so,
where it ends here (or that it remains open past this chunk too).

CURRENT STATE:
- Open paragraph carried from previous chunk: {yes/no}
  {If yes: it began at absolute offset {x}; its last known text ends with:
   "...{tail}"}
- Paragraphs already captured in this chunk: {n}; most recent ended at
  offset {b}.

YOUR TASK:
{carry case: Determine whether the start of this chunk continues the open
 paragraph; if it continues, report where it ENDS in this chunk, or that it
 remains open at this chunk's end.}
{normal case: Find the NEXT paragraph starting after offset {b}: report its
 start, and its end if within this chunk, or that it remains open at the
 chunk's end.}

CHUNK TEXT:
{chunk_text}

Return ONLY valid JSON:
{"found": true,
 "order": {n+1},
 "continues_previous": true|false,
 "start": <char offset, or null if continuing the carried paragraph>,
 "end": <char offset, or null if open at chunk end>,
 "open_at_chunk_end": true|false}
or {"found": false}
```

Same 5×-YES/NO order and boundary validations as the other detectors. This implements your realistic expectation directly: paragraphs rarely span two chunks, but the last paragraph of a chunk is very often cut — so the open-paragraph carry, end-confirmation-or-new-detection in the next chunk, is the primary mechanism, matching sections exactly.

**On splitting paragraph detection into two paths — yes, and your hierarchy argument is the reason.** Your gradient is: sentences are the atomic unit any model of any size can capture reliably 1x1 (small spans, local judgment); section *starts* are next (a marker is a local pattern); paragraphs demand more holistic judgment (where does a point end?); documents demand the most (whole-corpus continuity). So sentences prepare everything for paragraphs in the most constrained scenario, the way section identification prepares for documents. Therefore:

- **Path 1 paragraphs:** prompt-based 1x1 detection as above — LLM-capable models handle it, and it feeds Phase 2's immediate linking.
- **Path 2 paragraphs:** no per-chunk paragraph prompting at all. Paragraphs are **constructed in Phase 4** from the sentence graph: Stage 1 accumulates context pools over grammar/coreference/dependency edges in sentence order; Stage 2 evaluates structural continuity between consecutive pools; boundaries promote pools to ParagraphNodes. Paragraphs become evidence-derived, which is precisely what a constrained SLM cannot be trusted to judge per-chunk but the graph can support after grammar exists.

**Your embedded question — "does this overlap with sentences if sentence detection isn't working on detected paragraphs? what ensures nothing was missed?"** No overlap, and the guarantee comes from the layering: **sentence detection runs on the chunk directly and is span-exhaustive over it** (the 5×-NO completion check certifies that no uncaptured sentence remains anywhere in the chunk). Paragraph detection never gates sentence detection — sentences are tied to paragraphs afterward by span containment, not discovered through them. So even if a paragraph boundary is wrong or a paragraph is missed entirely, every sentence still exists as a node; a paragraph error can misgroup sentences but can never lose one. That is the "expected behavior" you were reaching for: completeness is anchored at the sentence layer, grouping quality at the paragraph layer, and the two failure modes are independent.

**Interim tying (both confirmed):** sentences → paragraph nodes; modality references → sentence node if contained, else paragraph node; paragraph nodes → the open section node, even though sections aren't yet refined — acceptable, as you stated, because refinement happens on graphs later without re-detection.

## F — The processing flow, reconciled

Your flow is right, with three alignment notes that make it internally consistent:

**Path identities.** Path 1 is the **deconstructor**: top-down within the chunk — section state → paragraphs 1x1 → sentences 1x1 within them → clean/correct → modality spans — natural for models that can hold the whole chunk's structure at once, and (per your G conclusion) the native shape for an OMEX parser, which emits the entire chunk graph in one non-autoregressive pass, collapsing every granular loop into one call plus cheap verification. Path 2 is the **constructor**: bottom-up — per chunk only section-state tracking, sentences 1x1, and modality spans 1x1; everything structural above sentences is assembled later from graphs. Both paths are retained for SLMs, LLMs, and OMEX; Path 2 additionally serves as the bootstrap/verification path that manufactures OMEX's training data.

**Ordering dependency (the one correction to the chart).** For Path 2, Phase 3 (grammar extraction, graph-native, per-SentenceNode + pairwise cross-sentence) **must precede** Phase 4, because Stage 1's context pools are walked over grammar, coreference, and dependency edges — those edges must exist first. Your chart already lists Phase 3 before Phase 4, so the sequence is correct; the note is that Phase 2's "cross-chunk aggregation" items marked (Path 1) genuinely belong to Path 1 only, and for Path 2 the aggregation of structure *is* Phase 4 — your line "Phase 2 now does stage 1 as well as stage 2" resolves cleanly as: the Stage 1/Stage 2 methodology is the aggregation mechanism for Path 2, and it executes in the Phase-4 slot, after grammar. One phase list, two routes through it.

**Stage 2's role for sections, stated precisely (your correction, locked in).** Stage 2 does **not** identify section elements — the per-chunk section tracker already found the markers, because sections are the formalized, format-bearing boundary layer. Stage 2, always operating in order (sentence order → pool order), does two things around sections: it validates whether accumulated content pools are *within the bounds* of a detected section element (continuation of the current/prior section, grouped correctly) versus a broken run-off that only superficially resembled a section; and it performs boundary evaluation for the structures that have no reliable surface markers — paragraphs first, then (future) documents, where the whole graph's relationship evidence is what makes a *true* document break distinguishable from a formatting hiccup without context explosion. Element detection is surface work (Phase 1); containment/continuity validation and unmarked-boundary promotion are evidence work (Phase 4). Documents remain untracked until that machinery exists — as you specified.

## G — OMEX: the eleven questions, and the name

The corrected README and docs above are the formal deliverable; here are the direct answers.

**G.1 — Would this yield a model with true semantic understanding of grammar, that actually understands language?** It yields something no LLM has: **explicit, inspectable, native structural competence**. Grammar is a closed system, and closed systems are masterable by small models — existing neural parsers already demonstrate near-human-agreement accuracy on structure at a fraction of LLM size, so the feasibility is not speculative. Two honest qualifications. First, LLMs do encode grammar — probing research shows syntax is recoverable from their internal representations — so the difference OMEX makes is *explicitness, separability, and auditability*, not presence versus absence. Second, "understands language" in full means structure **plus** meaning grounded in knowledge and context. OMEX deliberately splits that: the model natively understands structure; knowledge is referenced, not contained. So the accurate claim is: the *model* truly understands grammar; the *system* (parser + ZSEI traversal + AMT) is what understands language — understanding becomes a property of the composed architecture, with provenance, rather than an opaque property of one network. That's arguably a stronger position than the LLM's, because every step of it can be checked.

**G.2 — Grammar correction: in the grammar model or separate?** Same model, separate **head**, shared encoder. Detection and correction share nearly all representation — you must parse to correct — and your training data pairs them intrinsically (every SentenceNode carries original + corrected). Tagging-based correction (per-token edit operations) is proven to work as a lightweight head and decodes in the same single pass. A separate correction model would double inference and forfeit the shared representation for no benefit. Keep it a head so correction can be switched off when only structure is needed — and benchmark the joint-versus-separate variants anyway, consistent with your two-path philosophy; the joint model is the expected winner.

**G.3 — Knowledge detection: in the grammar model or separate?** Split by function. **Detection** (does this sentence/span revolve around a knowledgeable topic, and of what coarse kind?) is cheap, local, and stable — a head on the grammar model. **Linking** (which exact ZSEI path?) is a different problem with a different lifecycle — it depends on the contents of a growing knowledge graph — and belongs in a separate model.

**G.4 — Two-stage knowledge detection, general first, then a tying model?** Yes — your instinct matches the proven decomposition (mention detection → candidate generation → disambiguation). Stage A, inside the grammar model: knowledge-span head, binary plus coarse type. Stage B, the KnowledgeLinker: a bi-encoder mapping the span-in-context into ZSEI's embedding space, ANN search over container embeddings, optional zero-shot verification on low-margin candidates via ZSEI's existing pattern. The decisive advantage of the split is maintenance: when ZSEI grows, you refresh the linker's index or fine-tune the linker — the grammar model, the expensive one to retrain, is never touched. And because references ride on sentence nodes, AMT Stage 1 pools aggregate applicable knowledge automatically, so every promoted branch already carries its knowledge neighborhood — the property you called out.

**G.5 — Model per language?** Per-language **models** over a **universal schema**. Your GrammarNodeType taxonomy and properties are language-general in the same way Universal Dependencies is: one schema, per-language realization. Per-language models are smaller, faster, and at equal size outperform multilingual ones except for low-resource transfer; OMEX's local-first, load-only-what-you-need philosophy points the same way. A tiny language-ID router selects the model; AMT and ZSEI see one schema regardless. Multilingual variants remain useful as a bootstrap for low-resource languages, later distilled per-language. Verify empirically, as you said — but this is the expected outcome.

**G.6 — Do per-modality graph models yield the same natural understanding?** For **code and math: yes, and likely better** — their grammars are strictly formal and fully closed; an AST-native code model emitting your CodeEdgeType layer (Calls, DataFlows, ImplementsPattern) is the OMEX thesis in its purest form. For **image/audio/video: partially, with an honest boundary.** The *relational* layers of your modality graphs — SpatialRelationship, TemporalChain, ObjectInteraction, prosody-event structure — are learnable graph targets and the OMEX approach applies to them. But the *perception* step (pixels → objects, waveform → events) is not a closed rule system; it requires perceptual models of real size. So: structure extraction becomes native per modality; perception does not. Design consequence: perceptual pipelines pair a perception front-end with an OMEX relational model, while code/math need no front-end at all.

**G.7 — Fixed vs expandable matrices; run models together or sequentially?** Facts first: standard trained transformers (all SLMs/LLMs, and neural NLP models before them) have **fixed** weight matrices post-training. The known expansion mechanisms are: MoE (add experts — requires training), adapters/LoRA (bolt-on low-rank matrices — the one well-behaved additive expansion), progressive growth (Net2Net-style — a training technique, not runtime), and weight merging (soups/TIES — same-architecture fine-tunes only, brittle). What OMEX should do: **do not chase runtime weight expansion between heterogeneous models** — it's research-grade instability. OMEX's answer to "running together" is threefold and already in the format: (1) shared-encoder multi-head — genuinely one model executing many tasks in one pass; (2) adapter composition — a backbone "expanded" by domain adapters at load time, the legitimate version of what you're describing; (3) sequential graph pipelines — models exchange **typed graphs**, not hidden states, which is OMEX's actual innovation: the graph is the interoperability layer that weight surgery was trying to be. Sequential-with-overlap (pipeline parallelism across models) recovers most of the throughput that fused execution would give, without the fragility.

**G.8 — The training-generation prompt.** Delivered in full in OMEX-TRAINING.md §2, with the validation rules (§3) and the scaling/curriculum scheme (§4): one self-contained master prompt producing, per topic, the source text (with controllable error injection), the ordered sentence decomposition with corrections and spans, one complete grammar tree per sentence over the full taxonomy with properties, cross-sentence relationships, coreference chains, modality spans, knowledge references with topic paths, section/paragraph structure, AMT candidate units, and a knowledge-graph fragment — all in one machine-parseable JSON object, scalable to hundreds of thousands of examples via topic × register × error-density × seed, with each example serving parser and realizer training simultaneously.

**G.9 — CPU/GPU execution insights.** Full treatment in OMEX-EXECUTION.md; the ranked essentials: (1) **Non-autoregressive decoding is the whole game** — one forward pass per sentence versus hundreds of decode steps for LLM-emitted JSON; no kernel optimization approaches this. (2) **Batch sentences** — grammar extraction is embarrassingly parallel once order is fixed; weights are read once per batch, which is what a bandwidth-bound small model needs. (3) **Everything resident** — the family is <1 GB int8; no offloading logic should exist. (4) CPU: bandwidth-bound math (≈ bandwidth ÷ bytes/pass → hundreds of sentence-passes/s on ordinary DDR5, single-stream), int8/int4 quantization (classification heads tolerate it well), SIMD integer kernels, parallelize across batch items not within layers, NUMA-pin. (5) GPU: tiny models are kernel-launch-bound — CUDA Graphs, fused ops, large batches; keep tokenized batches device-resident; transfer graphs, not activations. (6) Multi-model: shared encoder amortizes; two-lane CPU/GPU pipeline (ZSEI traversal and ANN on CPU concurrent with encoder passes on GPU); cascade escalation to a big model only on low-confidence outputs. (7) Honest Amdahl note: until OMEX models replace the pipeline-9 calls, the LLM bootstrap dominates everything — the highest-value optimization is finishing the swap.

**G.10 — How does a model respond like an LLM after the decomposition?** By your own formulation, which is exactly right: **writing is grammar traversal in reverse**, and grammar sits on both ends — parser inbound, realizer outbound. The full loop: Input Text → Grammar Graph → (Stage 1/2 over graphs) → AMT/Intent Graph → Decision & task execution → **Response Graph** → Grammar Graph → Output Text. The Response Graph is assembled by the orchestrator, not imagined by a model: its content nodes come from task/step outputs, its structure from the AMT branch being answered, its knowledge content from resolved ZSEI references, its tone from VoiceIdentity properties attached as graph node properties. The **Realizer** then renders it — trained on the inverse pairs of the parser's corpus (the same data, reversed), decoding within the grammar schema so output is well-formed by construction. Task-checkpoint responses fall out naturally: an AMT branch reaching a completion or clarification checkpoint triggers Response-Graph assembly for that branch's neighborhood — evidence, results, and applicable knowledge are already aggregated on it. And this surpasses the LLM's response mechanism in one measurable way: the realizer **cannot hallucinate** — it renders only what the graph contains, and everything the graph contains has provenance (a task result, a ZSEI container, a grammar node with spans). Grammar competence from the model, knowledge from ZSEI, structure from the AMT — the "LLM-like response" is a property of the composed system, which is the entire point of breaking the LLM apart NLP-style at a level NLP could never reach.

**G.11 — ZSEI with OMEX: what to store, what to hold in memory.** Full treatment in OMEX-ZSEI-INTEGRATION.md; the operating summary: **always in RAM** — all OMEX weights (small, fixed), the ZSEI global mmap (OS-page-cache-managed, O(1) structural hops), the KnowledgeLinker's HNSW index (the latency-critical path), and the L1/L2 caches; **on demand** — per-container local JSON and knowledge-fragment neighborhoods, prefetched the moment the knowledge-span head fires so disk I/O overlaps model compute; **write-behind** — runtime-emitted graph containers and hook-inferred edges, batched per chunk so parsing never blocks on storage. Batch all span resolutions per chunk into one ANN pass plus one grouped ZSEI fetch; reuse the shared encoder's representations as the linker's span embeddings rather than running a separate embedding model; key the resolution cache on normalized span hashes with TTL invalidation, and keep ZSEI's L4 zero-shot verification cache for low-margin links. CPU/GPU split: ZSEI traversal is pointer-chasing CPU work that runs concurrently with GPU encoder passes — two lanes, joined at chunk boundaries. And the structural payoff of the whole design: because OMEX models hold no knowledge, **RAM scales with the working set, not the corpus** — ZSEI can grow to billions of containers while device memory stays flat.

**Is OMEX still a fitting name?** Yes — and it fits better now than it did before the corrections. **Omni**: across modalities (text, code, math, and the relational layers of the perceptual ones), across languages (one universal schema, per-language models), and across scales (the same format from a phone-resident parser to a workstation family). **Execution**: the format remains prompt-first and executable — a prompt enters, graphs flow through composed models, a response graph executes into text; the graph contracts in `io/` are literally execution interfaces. **Format**: it is now genuinely *the* canonical representation — of the model, of its schemas, and of the graphs that are the system's interchange currency. The name survives the redesign because the redesign removed the parts that weren't OMEX (embedded optimizers, hardware profiles) and kept the part that always was: a universal format for executing meaning.

---

Everything requested is above: the rewritten README with all removals applied and the corrected identity, four supporting technical documents, and A–G answered in full — including the redesigned prompts for B, C, D, and E, the flow reconciliation for F, and all eleven OMEX questions plus the name. If you want any single piece expanded to greater depth next — the master training prompt into a full generation-tool spec, the Realizer's decoding constraints, or the Phase-4 Stage 1/Stage 2 traversal written as concrete prompt sequences like B–E — point at it and we continue from there.
