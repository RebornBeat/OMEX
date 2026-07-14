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
