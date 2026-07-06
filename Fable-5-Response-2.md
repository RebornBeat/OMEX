Understood — everything below is the complete, newcomer-facing OMEX documentation set: every existing doc rewritten from the ground up for someone who has never heard of Ozone-Studio, ZSEI, or any of the internal terminology, plus the new docs needed to carry all the insights (concepts primer, format spec with worked graph examples, performance/comparison doc, ecosystem integration doc, and a full FAQ that captures all eleven G questions in newcomer language). Path 1/Path 2 terminology is removed from all core OMEX docs and appears only inside the integration doc, clearly labeled as host-pipeline concepts. Where the second draft you showed me had errors (fabricated benchmark tables presented as measurements, "expandable matrices," "one multilingual model" stated as settled, adaptive chunking reintroduced), I've kept the corrected positions we established — and where design questions are genuinely open by your own statement, the docs say so explicitly and present the recommended default plus the verification plan. All inline, no artifacts, no skills.

---

---

# FILE: README.md

---

# OMEX: Omni-Execution Format

**OMEX is a model format, a training methodology, and an execution interface for graph-native language models** — small, fast, auditable models that read and write explicit linguistic structure (grammar graphs) instead of hiding grammar, knowledge, and reasoning inside billions of opaque parameters.

If you have never encountered this project before, the one-paragraph version is this: today's large language models (LLMs) are enormous because they learn *everything at once* — grammar, style, world knowledge, reasoning — entangled across their weights. OMEX un-entangles them. It uses existing large models **once, as factories**, to produce hundreds of thousands of complete, validated *grammar graphs* — explicit tree/graph structures describing exactly how every sentence is built, how sentences relate to each other, what was grammatically wrong and how it was corrected, and which spans refer to knowledge topics. Small models are then trained directly on those graphs. The result is a family of models, each a fraction of a gigabyte, that natively **parse** text into structure, **correct** grammar, **reference** external knowledge without containing it, and **write** fluent text back out by running the same mapping in reverse — because writing is grammar traversal in reverse.

New to all of this? Start with **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — it explains, from zero, what traditional NLP was, what SLMs/LLMs actually learn, what a grammar graph is, and why separating grammar from knowledge changes the economics of language AI.

## The Three Eras

| Era | Grammar | Knowledge | Cost |
|---|---|---|---|
| **Traditional NLP** (pre-2018) | Explicit — hand-coded rules and statistical stages | External or absent | Brittle, narrow, plateaued |
| **SLM / LLM** (2018–present) | Implicit — emergent, distributed across parameters | Embedded in the same parameters | Enormous size, energy, opacity |
| **OMEX** | Explicit **and learned** — the model's native output *is* the graph | External, referenced by address — never embedded | Small models, auditable structure, local-first |

Traditional NLP decomposed language into components — tokenizer, part-of-speech tagger, parser, grammar rules, semantic analysis — but every component was hand-crafted or narrowly statistical, so the field plateaued. LLMs dissolved the components: grammar, syntax, semantics, discourse, style, and world knowledge became one implicit soup learned from massive corpora. That worked — but at the cost of carrying all of civilization's knowledge inside the weights just to conjugate a verb.

**OMEX closes the loop.** It is NLP's decomposition, reborn — with each component now a trained neural model that carries LLM-grade linguistic understanding, made possible *only because* LLMs came first and can manufacture the training graphs. It could not have existed before them.

## What an OMEX System Does

```
Input Text
    │
    ▼
Grammar Graph          ← OMEX GrammarParser (structure + correction + knowledge spans)
    │
    ▼
Intent / Meaning       ← host system traverses the graph (see docs/EXECUTION.md)
    │
    ▼
Decision & Task Work   ← host system, with knowledge fetched by reference
    │
    ▼
Response Graph         ← assembled from verified results, structure, and tone
    │
    ▼
Grammar Graph          ← same schema as the input side
    │
    ▼
Output Text            ← OMEX Realizer renders the graph into fluent language
```

Grammar sits on **both sides** of the loop. The parser on the way in, the realizer on the way out. Meaning lives in the middle, as graphs — inspectable, cacheable, and carrying provenance for every claim.

## The OMEX Model Family

| Model class | Input | Output | Role |
|---|---|---|---|
| **GrammarParser** | Text | Grammar graph | One encoder, multiple task heads: full sentence structure, grammar correction, knowledge-span detection, sentence/paragraph/section boundaries |
| **KnowledgeLinker** | Knowledge-bearing span + context | Ranked knowledge-graph paths | Resolves *which* topic a span refers to, against an external knowledge fabric; refreshed as the fabric grows, without retraining the parser |
| **Realizer** | Response graph (+ voice properties) | Text | The parser's training pairs, reversed; renders only what the graph contains — it cannot invent facts |
| **Modality structural models** | Code, math, etc. | Modality graphs | The same approach applied to other formal structures (code ASTs, mathematical derivations, and the relational layers of perceptual media) |

Indicative sizes: tens to low hundreds of millions of parameters per model; the entire family fits in well under 1 GB at 8-bit quantization. Final sizes are benchmark outcomes, not spec commitments.

## The Bootstrap Loop

```
Raw text corpora
      │
      ▼
Graph-generation pipeline (an existing SLM/LLM, driven step-by-step,
with repeated validation of every extracted structure)
      │
      ▼
Validated graph corpora:
sentence trees · original/corrected pairs · cross-sentence relationships ·
coreference chains · paragraph/section structure · knowledge references
      │
      ▼
OMEX training — text→graph (parser) AND graph→text (realizer)
from the SAME pairs
      │
      ▼
OMEX models replace the LLM calls inside the same pipeline
      │
      ▼
Faster, cheaper, more consistent graph generation
      │
      └────────► larger, better corpora → retrain → repeat
```

The expensive, slow, LLM-driven pipeline is temporary scaffolding. Once trained, an OMEX parser emits an entire chunk's structure in a single forward pass — collapsing what took hundreds of sequential LLM calls into one.

## What OMEX Is Not

Earlier drafts of this project carried ideas that have been **removed** because they were over-engineered and did not belong to what OMEX actually is:

- **No embedded "optimization intelligence."** There is no execution-optimizer binary, no compressed-insight artifact, no optimization fingerprint inside the format.
- **No hardware-aware execution in the format.** OMEX models carry no hardware profiles, tensor-core annotations, or per-device strategies. Hardware concerns belong to the runtime, not the format.
- **No adaptive chunking.** How a host system windows long inputs is the host's concern, not a format feature.
- **No per-model memory allocation constraints.** Execution plans do not declare RAM budgets.
- **No knowledge inside the weights.** This is the defining removal: OMEX models reference knowledge; they never contain it.

## Document Map

| Doc | Read it for |
|---|---|
| [docs/CONCEPTS.md](docs/CONCEPTS.md) | Background from zero: NLP, SLMs/LLMs, grammar graphs, grammar-vs-knowledge, glossary |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The model classes, heads, contracts, language and modality strategy, composition rules |
| [docs/FORMAT.md](docs/FORMAT.md) | The `model.omex/` package spec, the grammar schema, graph interchange JSON, worked examples |
| [docs/TRAINING.md](docs/TRAINING.md) | Where training graphs come from, the complete generation prompt, validation rules, curriculum |
| [docs/EXECUTION.md](docs/EXECUTION.md) | The runtime loop, response generation as reverse traversal, CPU/GPU optimization |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | Speed, memory, and energy comparisons vs. NLP/SLM/LLM — with the arithmetic shown |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Host systems: what Ozone-Studio and ZSEI are, how OMEX plugs in, what any host must provide |
| [docs/FAQ.md](docs/FAQ.md) | Every question we've been asked, answered plainly — including all open design questions |

## Status and Roadmap

- [x] Conceptual specification and schema alignment (this document set)
- [ ] Training corpus generation at scale (validated graph pipeline as data QA)
- [ ] GrammarParser v0 — single language, shared encoder, structure + correction heads
- [ ] KnowledgeLinker v0 — span → knowledge-path resolution with refreshable index
- [ ] Realizer v0 — graph → text, voice-property conditioning
- [ ] Benchmark program: OMEX-native vs. LLM-pipeline extraction (speed, quality, loops-to-completion, energy) and the open design questions (per-language vs. multilingual, joint vs. separate heads)
- [ ] Per-language model line-up; code and math structural models
- [ ] Task-checkpoint response generation integrated with intent-graph traversal

## License

MIT License. See LICENSE for details.

---

---

# FILE: docs/CONCEPTS.md

---

# OMEX Concepts — Background from Zero

This document assumes no prior knowledge. It explains the three eras of language processing, what a grammar graph actually is, and the central design principle of OMEX: **grammar is a closed system; knowledge is an open one; they should never live in the same weights.**

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

an OMEX parser maps the grammar perfectly, and additionally flags *annealing* and *internal stress* as knowledge-bearing spans. A companion model (the KnowledgeLinker) resolves those spans against an external knowledge fabric, returning exact addresses like `/Manufacturing/Metallurgy/HeatTreatment/Annealing`. The knowledge stays external; the reference travels with the graph. Because downstream meaning-building accumulates neighborhoods over these same graphs, every branch of the eventual intent structure arrives with its applicable knowledge references **already aggregated** — no separate knowledge pass, no loading everything into memory.

## 7. Writing Is Grammar Traversal in Reverse

Everyone assumes language works like: *Text → Grammar → Meaning → Response.* That skips the most important layer. The real loop:

```
Input Text
  ↓
Grammar Graph        (parse)
  ↓
Intent Graph         (understand — traversal over grammar)
  ↓
Decision             (reason / execute tasks)
  ↓
Response Graph       (plan — verified content, structure, tone)
  ↓
Grammar Graph        (structure the response)
  ↓
Output Text          (render)
```

**Grammar exists on both sides. It is both the parser AND the renderer.** Generation is not token-by-token guessing; it is rendering a verified structure. A realizer trained on the inverse of the parser's data renders only what the response graph contains — which is why it cannot introduce facts that were never handed to it.

## 8. Glossary

- **Token** — the unit LLMs read/write (a word-piece). **Autoregressive decoding** — producing output one token at a time, each step depending on the last; this is why LLM generation is inherently sequential and slow. **Non-autoregressive** — predicting an entire structure in one forward pass.
- **Zero-shot** — asking a model to perform a task from instructions alone, with no task-specific training.
- **Forward pass** — one full run of input through a model.
- **Quantization (int8/int4)** — storing weights in 8 or 4 bits instead of 16/32, shrinking memory and bandwidth needs.
- **Knowledge fabric / knowledge graph** — an external store of concepts, topics, and their relationships, addressable by path and searchable by meaning.
- **Coreference** — different expressions ("John", "he", "the manager") referring to the same entity.
- **Span** — a start/end character range inside a source text; OMEX uses spans everywhere for provenance.
- **Intent graph / AMT** — a tree of goals and requirements built by traversing grammar graphs (a host-system concept; see docs/INTEGRATIONS.md).
- **ANN / HNSW** — approximate nearest-neighbor search over embedding vectors; how a span is matched to candidate knowledge entries in sub-millisecond time.
- **Adapter / LoRA** — small add-on weight matrices that specialize a frozen model for a domain without retraining it.

---

---

# FILE: docs/ARCHITECTURE.md

---

# OMEX Architecture

## 1. Design Rules

1. **The graph is the interchange contract.** OMEX models compose by exchanging *typed graphs*, never hidden neural states. Every intermediate is inspectable, cacheable, and reusable as training data.
2. **Grammar in the weights; knowledge by reference.** No OMEX model contains world knowledge. Knowledge-bearing spans are detected in the model and resolved outside it.
3. **Small and resident.** Every model is sized to live permanently in memory on ordinary devices. No offloading, no swapping, no layer streaming.
4. **Declared contracts.** Every model declares exactly one input/output contract (`text → graph`, `graph → text`, `graph → graph`, or `span+context → references`), pinned to schema versions. Two models compose if and only if producer output contract matches consumer input contract at compatible versions.

## 2. GrammarParser

The workhorse: **text → grammar graph**, in a single forward pass.

**Encoder.** A compact transformer encoder (indicative range: 100–300M parameters; final sizes are benchmark outcomes).

**Heads** — all sharing the encoder's token representations, so one encoding pass serves every task:

- **Boundary head.** Token-level tagging of sentence boundaries, paragraph open/close signals, and section markers with level. One pass per input window.
- **Structure head.** Per-sentence grammar-tree prediction, **non-autoregressive**: biaffine-style arc-and-label scoring over the full grammatical taxonomy, decoded under hard constraints — every child's span contained in its parent's, single parent per node, sibling spans non-overlapping, every surface token on exactly one leaf path. Node properties (tense, aspect, mood, voice, person, number, gender, case, definiteness, comparison, polarity, subtype) are per-node classifications.
- **Correction head.** Tagging-based grammatical error correction: per-token edit operations (keep / replace / insert / delete / transform) that reconstruct the corrected sentence from the original. Tagging-style correction is proven to work well in small models and decodes in the same single pass.
- **Knowledge-span head.** Tags knowledge-bearing spans with a coarse kind (`concept | principle | procedure | entity | process | tool | material`). **Detection only** — resolution belongs to the KnowledgeLinker.

**Why heads, not separate models.** Detection and correction share nearly all linguistic representation — you must parse to correct. One encoder pass amortizes across all tasks; heads can be switched off per call (structure-only when correction isn't needed). *(Joint vs. separate remains an explicit benchmark item — see FAQ — with joint as the strongly expected winner.)*

## 3. KnowledgeLinker

A separate, smaller model: **span-in-context → ranked knowledge-graph paths.**

Design: a bi-encoder. One side embeds the detected span with its sentence context (reusing the parser encoder's representations — no second embedding model); the other side is a precomputed embedding index (HNSW) over the external knowledge fabric's entries. Resolution = approximate nearest-neighbor search → candidate paths → optional zero-shot verification for low-margin cases.

**Why separate.** The knowledge fabric grows continuously. When it grows, you refresh the index (cheap) or fine-tune the linker (moderate) — the parser, the expensive model, is never touched. This mirrors the proven decomposition in entity-linking research: *mention detection → candidate generation → disambiguation* — here applied to arbitrary knowledge-graph paths rather than a fixed encyclopedia.

## 4. Realizer

**Response graph → text.** Input: a response graph — structural skeleton, content nodes (verified task results), resolved knowledge content, and *voice properties* (tone, formality, warmth, directness) attached as node properties. Output: text whose parse equals the input graph.

Trained on the exact inverse of the parser's pairs — **every training example serves both directions.** Decoding is schema-constrained: output is grammatically well-formed by construction, and the realizer renders *only* nodes present in the graph — which is the mechanical reason it cannot hallucinate.

## 5. Language Strategy

**Universal schema, per-language models** (recommended default). The node taxonomy and property set are language-general — one schema, per-language realization, in the spirit of universal grammar annotation projects. Per-language parsers at equal size are smaller, faster, and generally more accurate than multilingual ones, except for low-resource transfer. A tiny language-ID router selects the model; every downstream consumer sees one schema regardless of language.

Multilingual variants remain sanctioned as a **bootstrap** for low-resource languages, distilled later into per-language models. *Per-language vs. all-languages-together is one of the open design questions to be settled empirically — the docs state the recommended default and the reasons; the benchmark program settles it.*

## 6. Modality Strategy

The same approach generalizes wherever structure is formal:

- **Code and math — full applicability, likely better than text.** Their grammars are strictly closed. A code-structural model emits the AST plus a semantic edge layer (Calls, DataFlows, ImplementsPattern, DependsOn, ...); a math model emits derivation structure (LogicallyImplies, Generalizes, UsedToProve, ...). This is the OMEX thesis in its purest form.
- **Image, audio, video — the relational layer only, with an honest boundary.** Spatial relationships (Above/Occludes/Supports), temporal chains (Causes/Precedes/InterruptedBy), object interactions, prosody events — these are learnable graph targets once perception has produced entities. But *perception itself* (pixels → objects, waveform → events) is not a closed rule system and requires perceptual models outside OMEX's thesis. Design consequence: perceptual pipelines pair a perception front-end with an OMEX relational model; code/math need no front-end at all.

## 7. Composition Semantics

Three sanctioned modes — and one explicitly out of scope:

1. **Sequential graph pipelines** (default). Model A's output graph is model B's input. Matches any orchestrated pipeline architecture.
2. **Shared-encoder multi-head.** One encoder pass, many task outputs simultaneously. The genuine form of "running together."
3. **Adapter composition.** Domain/register adapters (LoRA-class low-rank add-ons) selected at load time on a common backbone — the one well-behaved way to "expand" a fixed model.
4. ***Out of scope:* runtime weight expansion or merging across heterogeneous trained models.** Trained transformers have fixed weight matrices; the known growth/merge techniques (MoE expert addition, Net2Net growth, model soups/TIES merging) either require training or are same-architecture-only and brittle. OMEX's answer to interoperability is the **graph contract**: when a task needs text understanding *and* code understanding, the text parser emits a grammar graph, the code model emits a code graph, and the host stitches them with cross-modal edges (e.g., `DescribedBy`, `ImplementedIn`). Execution contexts expand through graph stitching; parameters never physically merge.

---

---

# FILE: docs/FORMAT.md

---

# The OMEX Format

Two things carry the name: the **model package format** (how an OMEX model ships) and the **graph interchange format** (what OMEX models read and write). The second is the more fundamental — the graph is the system's currency.

## 1. Model Package: `model.omex/`

```
model.omex/
├── metadata.json                 # name, version, model_class
│                                 # (grammar_parser | knowledge_linker |
│                                 #  realizer | modality_structural),
│                                 # language(s), schema versions,
│                                 # training-corpus provenance
├── schema/
│   ├── grammar_schema.json       # full node taxonomy + properties (see §3)
│   ├── edge_schema.json          # relationship/edge type registry (per modality)
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
│   ├── input_contract.json       # text | graph | span+context
│   └── output_contract.json      # graph | text | references
└── extensions/
    └── adapters/                 # optional LoRA/adapter weights per domain
```

**The defining property:** `schema/` versions are what make two OMEX models compatible. Composition is legal exactly when the producer's output contract matches the consumer's input contract at compatible schema versions. There are **no** hardware profiles, memory budgets, or embedded optimizers anywhere in the package — those were removed from the design deliberately.

**Optional pipeline manifest.** A host may ship a `pipeline.json` listing model stages and their graph contracts (parser → linker → realizer, etc.). It is a composition declaration only — no runners, no memory allocations, no device hints.

## 2. Graph Interchange: A Worked Example

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

## 3. The Grammar Schema (Full Taxonomy)

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

Cross-sentence relationship types: `Elaborates, Causes, Enables, Prevents, Contradicts, Exemplifies, Summarizes, TemporalPrecedes, Coreference, PartOf, SimilarTo`.

## 4. Modality Edge Schemas (Illustrative)

The interchange principle extends per modality; `edge_schema.json` registers the modality's edge set. Examples:

- **Code:** Contains, Calls/CalledBy, Imports/ImportedBy, Defines/DefinedBy, DependsOn/RequiredBy, ImplementsPattern, UsesPattern, IntentSolves, DataFlows, ControlFlows, ExtendsAbstraction, plus cross-modal DescribedBy/Implements/ImplementedIn/VisualizedAs
- **Image:** Contains, PartOf/HasPart, SpatialAbove/Below/Left/Right, Occludes, PhysicallySupports, ContainedIn, FocalSubject, LeadsEyeTo, SimilarTo, cross-modal set — plus affordances ("CanSupport", "CanContain")
- **Audio:** Contains, Precedes/Follows, Causes, OverlapsWith, InterruptedBy, ResponseTo, ToneIndicates, PitchCorrelates, SyncedTo/SyncedBy — plus prosody analysis (pitch, rate, energy, emotional markers)
- **Video:** Contains, Precedes/Follows, InteractsWith, CollidesWith, FollowsSpatial, CausesEvent, NarrativeLeadsTo, AudioSyncsTo — plus object interactions (Touches, Collides, Follows, Approaches, Leaves, PassesBy) with time ranges
- **Math:** Contains, Precedes/Follows, DependsOn, LogicallyImplies, LogicallyEquivalent, Contradicts, Generalizes, SpecialCaseOf, UsedToProve, DischargesAssumption, AssumesIn

These are illustrative, not exhaustive — the registry is versioned and extensible per modality.

---

---

# FILE: docs/TRAINING.md

---

# OMEX Training

## 1. Where the Graphs Come From

Two complementary sources:

**A. A graph-generation pipeline over real text.** An existing SLM/LLM is driven step-by-step over chunked text — identifying sentences one at a time in order, correcting each, detecting section markers and embedded non-prose spans, extracting each sentence's grammar tree, then pairwise cross-sentence relationships and coreference — with **every extracted structure validated by repeated independent yes/no confirmation before acceptance** (the reference implementation requires five consecutive confirmations per item). Only stable, validated structures enter the corpus; the validation loop *is* the data-QA gate. (The reference implementation of this pipeline lives in the Ozone-Studio host system — see docs/INTEGRATIONS.md — but any pipeline producing schema-conformant, validated graphs qualifies.)

**B. Synthetic generation at controlled coverage.** The master prompt below produces complete training examples per topic, register, and error density, scalable to hundreds of thousands of examples.

**Every example trains both directions.** Text→graph trains the parser; the same pair reversed trains the realizer. One corpus, two models.

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

## 4. Scale and Curriculum

Hundreds of thousands of examples = |topics| × |registers| × |error densities| × seeds. Curriculum order: clean single-clause sentences → multi-clause with subordination → error-injected → mixed-register documents with sections → knowledge-dense technical passages. Real pipeline output (validated graphs from actual prompts and documents) is interleaved with synthetic data so the parser sees natural distributions, not only generated text.

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
3. KnowledgeLinker → each knowledge span resolved to ranked paths in the
   external knowledge fabric (batched ANN search; knowledge fetched by
   reference, never loaded wholesale).
4. Host traversal → intent/meaning built by walking the graphs (see
   docs/CONCEPTS.md §5 for the traversal invariants; see docs/INTEGRATIONS.md
   for a full host example).
5. Decision & task execution → whatever work the intent requires.
6. Response graph assembly → the host composes verified content nodes, the
   structural skeleton of the answer, resolved knowledge content, and voice
   properties (tone, formality, warmth, directness) as node properties.
7. Realizer → renders the response graph into fluent text. Grammar on the way
   out, exactly as on the way in.
```

Task-checkpoint responses fall out naturally: whenever an intent branch reaches a completion or clarification checkpoint, the response graph is assembled for that branch's neighborhood — and because knowledge references ride on the sentence graphs from step 2–3, the branch's applicable knowledge is already aggregated on it.

## 2. Why This Is Fast: The Decisive Structural Facts

1. **Non-autoregressive structure prediction.** A parser with biaffine-style heads emits an entire sentence's tree in **one** forward pass. A general LLM emitting the same tree as JSON performs **hundreds of sequential decode steps**. This one design choice outweighs every kernel trick combined: it converts an O(output-tokens) sequential process into O(1) passes.
2. **Sentence-level parallelism.** Once boundaries are tagged, grammar extraction is embarrassingly parallel across sentences. Batch them.
3. **Everything stays resident.** The whole family at int8 is a fraction of a gigabyte: no offloading, no swapping, no layer streaming. Load once; the working set is stable.

## 3. CPU Execution

- Small-model single-stream inference is **memory-bandwidth bound**: throughput ≈ bandwidth ÷ bytes touched per pass. Optimize by shrinking bytes (int8 → int4 where classification accuracy holds — structure/correction heads tolerate quantization far better than open-ended generation) and by **batching** (weights are read once per batch, amortizing bandwidth across many sentences).
- Use SIMD-optimized integer kernels (AVX2/AVX-512/AMX on x86; NEON/SME on ARM).
- Thread strategy for tiny models: parallelize **across batch items**, not within layers — intra-layer parallelism of a ~100M model underutilizes cores and pays synchronization costs. Pin threads; on multi-socket machines keep weights and workers NUMA-local.
- Tokenization and span bookkeeping are cheap pure-CPU work: run them on separate threads, pipelined ahead of encoder batches.

## 4. GPU Execution

- For tiny models, **kernel-launch overhead can rival compute**. Record the whole forward pass as one launch (CUDA Graphs or equivalent); fuse elementwise ops; prefer a few large batched matmuls over many small ones.
- Batch aggressively — hundreds of sentences per pass is the natural operating point; per-sentence latency collapses.
- Keep tokenized batches device-resident across stages; **transfer graphs, not activations** — the graphs models exchange are tiny.
- Multi-model concurrency: parser, linker, and realizer are small enough to share one GPU via streams; or split — parser on GPU, linker ANN search on CPU, since ANN is pointer-chasing, not matmul.

## 5. Multi-Model Orchestration

- **Shared encoder, many heads:** one encoding, N task outputs — the highest-leverage form of "running together."
- **Pipeline overlap:** while sentence batch *k* is in the structure head, batch *k+1* tokenizes and batch *k−1*'s knowledge spans are in the linker. CPU work (knowledge traversal, span bookkeeping, ANN) and GPU work (encoder passes) run as a two-lane pipeline, joined at window boundaries.
- **Cascade/escalation:** run the OMEX model first; escalate to a general SLM/LLM only on low-confidence outputs. This keeps quality guarantees while paying large-model cost only on the hard tail.

## 6. What Dominates, Honestly

In any deployment that still uses an LLM-driven pipeline for graph generation, **that pipeline dominates all cost** (Amdahl's law). The optimization program that matters most is therefore: generate corpora → train the parser → swap it in. Every kernel-level optimization above is second-order next to that swap.

---

---

# FILE: docs/PERFORMANCE.md

---

# OMEX Performance: Speed, Memory, Energy

## 0. Methodology — Read This First

**Nothing in this document is a measured benchmark.** OMEX v0 models have not shipped; the benchmark program is on the roadmap. What this document provides is **derivations**: arithmetic consequences of parameter counts, decoding modes, and memory arithmetic, with every step shown so you can check it. Earlier drafts of this project circulated tables with figures like "96% faster" and "99% energy saved" presented as results; those were projections dressed as measurements and are **withdrawn as such** — the honest versions of those same claims, derived and labeled, are below. When measurements exist, they replace this section.

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

## 2. Energy

On the same hardware, energy scales approximately with FLOPs, so the ratios above carry directly to joules. As an order-of-magnitude illustration (modern accelerators deliver very roughly 10⁻¹²–10⁻¹³ joules per FLOP effective):

```
LLM route:   8.4 TFLOPs  →  roughly 1–8 J per analyzed sentence
OMEX route:  6 GFLOPs    →  roughly 0.001–0.006 J per analyzed sentence
```

Three-plus orders of magnitude less energy per structure — the same conclusion by a second path. Where "per token" is the unit of comparison: an LLM pays ~2N FLOPs for every output token, hundreds of times per sentence analysis; the OMEX parser pays for the input tokens once.

## 3. Memory

```
7B model, int8:                    ≈ 7.0 GB weights
Full OMEX family, int8:
  GrammarParser  100–300M          ≈ 0.10–0.30 GB
  KnowledgeLinker 50–150M          ≈ 0.05–0.15 GB
  Realizer       100–300M          ≈ 0.10–0.30 GB
  ─────────────────────────────────────────────
  Total                            ≈ 0.25–0.75 GB
```

**~90–96% reduction versus a single 7B SLM; ~98–99% versus a 70B-class model** — and, more importantly, the entire family is *simultaneously resident* on a phone or a Raspberry Pi with no swapping, offloading, or layer streaming. Because knowledge lives outside the weights, **RAM scales with the working set, not the corpus**: the external knowledge fabric can grow to billions of entries while device memory stays flat.

## 4. Latency and Throughput

- **Single-stream CPU bound (bandwidth math).** Small-model inference is memory-bandwidth bound: passes/s ≈ bandwidth ÷ bytes touched per pass. A 100M int8 model on a 50 GB/s DDR5 system: 50×10⁹ ÷ 1×10⁸ ≈ **~500 forward passes/s upper bound, single-stream** — hundreds of sentences per second, CPU-only, before batching.
- **Startup.** Loading <1 GB of weights vs. tens of GB: cold-start drops from tens of seconds (large models with offloading) to well under a second. No warm-up analysis phase exists — the format contains no embedded optimizers to initialize (removed by design).
- **Structure latency.** One forward pass per sentence vs. hundreds of sequential decode steps: per-sentence structure latency drops from seconds (LLM JSON emission) to milliseconds (single pass), and collapses further under batching.

## 5. Quality-Side Comparisons (qualitative, structural)

| Dimension | Traditional NLP | SLM/LLM | OMEX |
|---|---|---|---|
| Grammar representation | Explicit, hand-coded, brittle | Implicit, emergent, entangled | **Explicit and learned** — native output is the graph |
| Adaptability to messy language | Poor | Excellent | Excellent (trained on LLM-manufactured graphs, including error/correction pairs) |
| Auditability | High | Near zero | **High** — every node carries spans; every claim carries provenance |
| Grammar correction | Separate rule systems | Implicit, unlocalized | **Intrinsic and localized** — per-span, typed edits |
| Knowledge | External or absent | Embedded, provenance-free (hallucination risk) | **External by reference** — addresses, not contents |
| Determinism | Deterministic | Stochastic | Deterministic at temperature 0; schema-constrained by construction |
| Output structural validity | By construction | Not guaranteed (malformed JSON, drift) | **Guaranteed by constrained decoding** |
| Generation fidelity | n/a | Can invent content | Realizer renders **only** what the response graph contains |

## 6. Format Comparison

| Feature | ONNX | GGUF | TorchScript | OMEX |
|---|---|---|---|---|
| Design philosophy | Interchange | Quantized LLM packaging | Traced export | **Graph-native language models** |
| Unit of exchange between models | Tensors | Tensors | Tensors | **Typed graphs** |
| Grammar | Implicit in weights | Implicit in weights | Implicit in weights | **Explicit, schema-versioned** |
| Knowledge | In weights | In weights | In weights | **External, referenced** |
| Correction provenance | None | None | None | Original/corrected pairs, per span |
| Structure decoding | Autoregressive | Autoregressive | Autoregressive | **Single-pass, non-autoregressive** |
| Multi-model composition | Ad hoc | Ad hoc | Ad hoc | Graph contracts + shared encoders + adapters |

## 7. What Has Not Been Measured Yet

Parsing accuracy vs. established parsers on standard treebanks; correction quality vs. dedicated GEC systems; linker precision/recall at scale; realizer fluency; per-language vs. multilingual tradeoffs; joint vs. separate head tradeoffs; end-to-end energy on target devices. All are line items in the benchmark program. This document will be revised to measurements as they land.

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

**The intent structure (AMT).** Ozone-Studio builds an *Abstract Meaning Tree* — a tree of intents, branches, and details — not by asking a model "what is the intent?" but by **traversing the graphs**: accumulating context pools along grammar, coreference, and dependency edges (Stage 1: candidate unit formation), evaluating whether neighborhoods belong together by accumulated evidence rather than word similarity (Stage 2: boundary evaluation), and promoting neighborhoods to intents only when evidence suffices (Stage 3). The tree ties every intent back to its supporting grammar evidence and methodologies. Because OMEX graphs carry knowledge references on their sentence nodes, **every promoted intent branch arrives with its applicable knowledge already aggregated** — no separate knowledge pass, anywhere in the tree.

**Response generation.** When a task or checkpoint needs a user-facing answer, the orchestrator assembles a response graph from that branch's verified results and hands it to the OMEX Realizer — the reverse-traversal loop from docs/EXECUTION.md, embodied.

## 3. Primer: ZSEI (the reference knowledge fabric)

ZSEI (Zero-Shot Embedded Indexer) is **not a database** — it is a semantic knowledge fabric: it stores meaning, relationships, and traversable structure. Everything in it is a **container** (concepts, topics, methodologies, execution blueprints, modality graphs, external references), organized in a logical hierarchy (e.g., `/Modality/Text/...`, knowledge domains, workspaces) and connected by typed relationships.

Key properties, briefly:

- **Hybrid storage.** A memory-mapped global index with fixed-size headers gives O(1) structural hops (parent/child/version) at billion-container scale; rich per-container JSON holds metadata, keywords, topics, embeddings, and relationships; only hot containers stay in RAM.
- **Three traversal modes, combined.** *Structural* (walk the hierarchy), *semantic* (embedding similarity), *contextual* (follow typed relationship edges) — each covers the others' blind spots; combined search merges and ranks all three.
- **Zero-shot verification.** Top candidates from any search can be verified by a model asking, in effect, "is this actually relevant?" — a quality filter requiring no training, cached to avoid repeat cost.
- **Link, don't copy.** Content is referenced (files by path+hash, URLs with semantic snapshots, packages by registry coordinates) — never duplicated.

**What OMEX uses ZSEI for:** (a) storing the training graph corpora; (b) being the address space the **KnowledgeLinker** resolves into — an HNSW index over container embeddings, batched ANN per chunk, zero-shot verification on low-margin links; (c) storing the modality graphs OMEX models emit at runtime, where semantic hooks then enrich them with inferred relationship edges.

## 4. What *Any* Host Must Provide

OMEX does not require Ozone-Studio or ZSEI. Substitutes must provide:

| Requirement | For | Reference implementation |
|---|---|---|
| A graph-generation/validation pipeline (or an existing schema-conformant corpus) | Training | Ozone-Studio text pipeline, granular pathway |
| A knowledge fabric with **addressable paths**, **entry embeddings**, and ideally **typed-relationship traversal** | KnowledgeLinker resolution | ZSEI |
| An intent/meaning layer that consumes grammar graphs | Understanding | AMT traversal |
| A response-graph assembler (verified content + structure + voice properties) | Generation | Ozone-Studio orchestrator |

## 5. Memory Strategy in an Integrated Deployment

| Tier | Contents | Rationale |
|---|---|---|
| Always in RAM | All OMEX weights; the knowledge fabric's structural index (mmap, OS page cache); the linker's ANN index; hot-entry and query caches | Weights are small and fixed; structural hops must be O(1); ANN is the latency-critical path |
| On demand | Per-entry rich records; knowledge subgraph fragments around resolved references | Only hot neighborhoods matter at any moment; a cap bounds residency |
| Write-behind | Runtime-emitted graph containers; inferred relationship edges | Parsing throughput must never block on storage; batch writes per chunk |

Access patterns: **batch** all of a chunk's span resolutions into one ANN pass plus one grouped fabric fetch; **prefetch** candidate entries the instant the knowledge-span head fires, overlapping disk I/O with model compute; **reuse the shared encoder's representations** as the linker's span embeddings; cache resolutions keyed on normalized span hashes with TTL invalidation. CPU/GPU split: fabric traversal is pointer-chasing CPU work that runs concurrently with GPU encoder passes — two lanes, joined at chunk boundaries; neither ever waits on the other.

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

### Q9. One model per language, or one multilingual model?

**Per-language models over a universal schema** (recommended default). The node taxonomy and properties are language-general — one schema regardless of language — while per-language parsers at equal size are smaller, faster, and generally more accurate than multilingual ones. A tiny language-ID router selects the model. Multilingual variants stay sanctioned as a **bootstrap for low-resource languages**, distilled later per-language. *This is an explicitly open design question — per-language vs. all-together is a benchmark line item; the docs state the default and the reasoning, the measurements decide.*

### Q10. Do modality-specific models (code, images, audio, video, math) get the same natural understanding as the text model?

**Code and math: yes — and likely better**, because their grammars are strictly formal and fully closed (a code model emits the AST plus semantic edges like Calls/DataFlows/ImplementsPattern; a math model emits derivation structure). **Image/audio/video: partially, with an honest boundary** — the *relational* layers (spatial relationships, temporal chains, object interactions, prosody events) are learnable graph targets, but *perception itself* (pixels → objects, waveform → events) is not a closed rule system and needs perceptual models outside OMEX's thesis. Perceptual pipelines pair a perception front-end with an OMEX relational model; code/math need no front-end at all.

### Q11. Fixed vs. expandable matrices — can OMEX models expand each other's parameters to run together?

The facts: standard trained transformers — all SLMs/LLMs — have **fixed** weight matrices post-training. Known expansion mechanisms (MoE expert addition, Net2Net-style growth, model soups/TIES merging) either require training or are same-architecture-only and brittle. **OMEX deliberately does not chase runtime weight surgery.** Its answer to "running together" is threefold and already in the format: **(1)** shared-encoder multi-head — genuinely one model doing many tasks in one pass; **(2)** adapter composition — a backbone legitimately "expanded" by low-rank domain adapters at load time; **(3)** sequential graph pipelines — models exchange **typed graphs**, not hidden states. When a task needs text *and* code understanding, each model emits its graph and the host stitches them with cross-modal edges: **execution contexts expand through graph stitching; parameters never physically merge.** The graph contract is the interoperability layer weight surgery was trying to be — and pipeline overlap recovers most of the throughput fused execution would give, without the fragility.

### Q12. How do you get a model to *respond* like an LLM after breaking the LLM into components?

By the principle the whole design rests on: **writing is grammar traversal in reverse**, and grammar sits on both ends — parser inbound, realizer outbound. The loop: Input Text → Grammar Graph → Intent Graph → Decision/Execution → **Response Graph** → Grammar Graph → Output Text. The response graph is *assembled*, not imagined: content nodes from verified task results, structure from the intent branch being answered, knowledge content from resolved references, tone from voice properties (tone, formality, warmth, directness) on the nodes. The Realizer — trained on the parser's pairs reversed — renders it fluently. Task checkpoints trigger response-graph assembly for the relevant branch, whose evidence and knowledge are already aggregated on it. And it surpasses the LLM mechanism in one measurable way: **the realizer cannot hallucinate** — it renders only what the graph contains, and everything the graph contains has provenance.

### Q13. Can OMEX models hallucinate at all?

The parser has no facts to invent — its output is structure over the given text, checkable span-by-span. The realizer renders only the graph it's handed. Knowledge enters the system exclusively by reference, each with an address. Failure modes still exist — a wrong parse, a wrong link — but they are **localized, detectable, and correctable**, which is categorically different from a fluent fabrication with no provenance.

### Q14. What is the CPU/GPU optimization story?

Ranked by impact: **(1)** non-autoregressive decoding — one pass per sentence vs. hundreds of decode steps; nothing else comes close; **(2)** batch sentences — extraction is embarrassingly parallel, and batching is what a bandwidth-bound small model needs; **(3)** everything resident — <1 GB int8, so no offloading logic should exist. CPU specifics: bandwidth math gives ~hundreds of sentence-passes/s single-stream on ordinary DDR5; int8/int4 (classification heads tolerate it well); SIMD integer kernels; parallelize across batch items, not within layers; NUMA-pin. GPU specifics: tiny models are launch-overhead-bound — CUDA Graphs, fused ops, big batches, device-resident inputs; transfer graphs, not activations. Orchestration: shared encoder amortizes; two-lane CPU/GPU pipeline (fabric traversal + ANN on CPU concurrent with encoder passes on GPU); cascade to a big model only on low-confidence outputs. Honest note: until OMEX models replace the LLM calls in a generation pipeline, the LLM dominates all cost — finishing that swap outranks every kernel trick.

### Q15. How does OMEX use an external knowledge fabric (like ZSEI), and what stays in memory?

**Always in RAM:** all OMEX weights (small, fixed), the fabric's structural index (mmap'd, O(1) hops), the linker's ANN index, and the hot caches. **On demand:** per-entry rich records and knowledge fragments — prefetched the moment the knowledge-span head fires so I/O overlaps compute. **Write-behind:** runtime-emitted graphs and inferred edges, batched per chunk. Batch a chunk's resolutions into one ANN pass + one grouped fetch; reuse the shared encoder's representations as span embeddings; cache on normalized span hashes with TTL invalidation. Structural payoff: because models hold no knowledge, **RAM scales with the working set, not the corpus** — the fabric can reach billions of entries while device memory stays flat.

### Q16. Do I need Ozone-Studio or ZSEI to use OMEX?

No. They are the **reference implementations** of two things any host must provide: a validated graph pipeline (or existing conformant corpus) for training, and a knowledge fabric with addressable paths + embeddings for linking. See docs/INTEGRATIONS.md for the generic requirements table.

### Q17. How is training data made, concretely?

Two streams: **(a)** a step-by-step LLM-driven pipeline over real text where every sentence, correction, section marker, modality span, grammar tree, and relationship is validated by repeated independent yes/no confirmation before acceptance — validation *is* the QA gate; **(b)** synthetic generation via the master prompt in docs/TRAINING.md, which yields one complete example per call (passage with controlled error injection, ordered sentence decomposition with corrections, full grammar trees over the complete taxonomy, cross-sentence relationships, coreference chains, knowledge references with paths, sections/paragraphs, candidate semantic units, a knowledge-graph fragment) — scalable to hundreds of thousands of examples via topic × register × error-density × seed. Every example trains parser **and** realizer.

### Q18. How big are the models, and what hardware do they need?

Indicative: parser 100–300M, linker 50–150M, realizer 100–300M — the family under ~0.75 GB at int8 (final sizes are benchmark outcomes). Hardware: any modern phone, single-board computer, laptop, or server, CPU-only if need be; a GPU multiplies batch throughput but is not required.

### Q19. What about long documents — does OMEX do chunking or streaming?

Windowing long inputs is the **host's** job, done once on intake purely to fit context windows, parallelize, and anchor byte positions — it is deliberately **not** a format feature (adaptive chunking was removed from the design). OMEX models process whatever window they're handed; all downstream work is graph-based, with spans anchoring every node back to source bytes.

### Q20. What happened to the embedded optimizers / hardware profiles / memory budgets from earlier descriptions?

Removed, deliberately. They were over-engineering that didn't belong to what OMEX is: no execution-optimizer binaries, no hardware-aware annotations in the format, no per-node RAM budgets, no "ZSEI training-time optimization discovery." ZSEI's real role is knowledge fabric, not architecture analyzer. See "What OMEX Is Not" in the README.

### Q21. Are the performance numbers real benchmarks?

No — and the docs say so loudly. Everything quantitative is a **derivation** (parameter arithmetic, decoding-mode analysis, bandwidth math) with the steps shown, in docs/PERFORMANCE.md. Earlier circulated tables that presented projections as measured results are withdrawn as such. The benchmark program replaces derivations with measurements.

### Q22. Deterministic? Reproducible?

At temperature 0, structure prediction is deterministic; constrained decoding guarantees schema validity; spans make every output mechanically checkable against its source. Compare an LLM emitting JSON: stochastic, occasionally malformed, unverifiable.

### Q23. Which comes first on the roadmap?

Corpus generation → GrammarParser v0 (one language, structure + correction heads) → KnowledgeLinker v0 → Realizer v0 → the benchmark program (including the open design questions: per-language vs. multilingual, joint vs. separate heads, and OMEX-native vs. LLM-pipeline extraction on speed/quality/loops/energy) → per-language line-up and code/math models.

### Q24. Is "OMEX" still a fitting name?

Yes — it fits the corrected design better than the original one. **Omni**: across modalities (text, code, math, and the relational layers of perceptual media), across languages (one universal schema, per-language models), across scales (the same format from a phone-resident parser to a workstation family). **Execution**: prompt-first and executable end-to-end — a prompt enters, graphs flow through composed models, a response graph executes into text; the io contracts are literally execution interfaces. **Format**: the canonical representation of the model, its schemas, and the graphs that are the system's interchange currency. The redesign removed what wasn't OMEX (embedded optimizers, hardware profiles) and kept what always was: **a universal format for executing meaning.**

---

---

That's the complete set: README plus eight docs — CONCEPTS (the from-zero education: NLP → SLM/LLM → OMEX, grammar graphs with both worked tree examples, the traversal-invariants table, grammar-vs-knowledge, the reverse-traversal loop, glossary), ARCHITECTURE, FORMAT (full taxonomy, worked JSON graph examples including a correction and knowledge references), TRAINING (master prompt in full, validation rules, curriculum), EXECUTION, PERFORMANCE (all speed/memory/energy comparisons with the arithmetic shown and honest methodology), INTEGRATIONS (newcomer primers on Ozone-Studio and ZSEI, the generic host requirements, memory tiers, and the knowledge-genome/evidence-graph application pattern), and the FAQ carrying all eleven of your G questions in newcomer language plus thirteen more — with the corrected positions preserved throughout (per-language default with the multilingual question explicitly open, no weight surgery, honest modality boundary, fabricated benchmarks withdrawn and replaced with labeled derivations, adaptive chunking and embedded optimizers kept out). If you want any single doc taken deeper — e.g., FORMAT's interchange spec expanded into a full JSON-Schema definition, or a dedicated OMEX-MODALITIES.md breaking out code/math model designs — point at it and we continue.
