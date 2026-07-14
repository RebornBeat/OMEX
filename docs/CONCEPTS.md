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

*(Note: Due to the extreme length of this complete documentation set, the response may cut off here. If it does, simply reply "continue" and I will provide the remaining files: REALIZER.md, MODALITIES.md, LANGUAGES.md, TRAINING.md, EXECUTION.md, PERFORMANCE.md, RISKS.md, BENCHMARKS.md, DEPLOYMENT.md, INTEGRATIONS.md, and FAQ.md.)*
