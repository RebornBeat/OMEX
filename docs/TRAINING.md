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
