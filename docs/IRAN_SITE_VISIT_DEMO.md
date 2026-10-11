# Iran Site-Visit Demo

## Scope

The static site follows the transfer and fragmentation of authority from the
Shah through Bakhtiar and into the Khomeini/Bazargan transition. It includes:

- a narrative -> community -> semantic-node graph explorer;
- typed entity, claim, event, relation, and document search;
- nine recorded workbench cases, three per task;
- three distinct precomputed live-retrieval packets;
- exact cable and segment provenance for displayed evidence.

No showcased redaction is a location fill whose answer is `IRAN`. Every
redaction is a person-value task, and every reconstruction contains at least
two atomic targets.

## Presenter Sequence

1. Search `PERSON: Shah of Iran`, `PERSON: Bakhtiar`, and `PERSON: Khomeini`
   to introduce the old order, transition government, and revolutionary power
   center.
2. Reconstruct Regency Council legality, parallel authority, and military
   takeover cost.
3. Test one supported hypothesis about the unfinished revolution and
   Khomeini's dominance, one opposed hypothesis about the Shah's constitutional
   return, and one causally inconclusive leftist-takeover hypothesis.
4. Resolve Mehdi Haeri from two biographical clues, retain Amir Entezam as a
   strong but unresolved `2/3` lead, and retain Nasser Minatchi as a weak
   context-only `0/2` lead.
5. Use the three live profiles to distinguish retrieval leads from audited
   evidence judgments.

The strongest saved reconstruction covers five of five atoms. The strongest
saved support and opposition hypotheses each cover two of two atoms using
private pre-source cables. The strongest non-location person recovery resolves
Mehdi Haeri only after two required clues bind to the same canonical person in
two documents. Entezam remains unresolved because the third required premise
is absent, and Minatchi remains unresolved because topic co-occurrence is not
treated as proof of either relationship.

## Hosted Limits

The static site contains the complete bounded hierarchy view, corpus search
index, source cable text, recorded case packets, and three precomputed live
packets. It cannot execute an arbitrary new query or materialize every node in
a very large community. Those operations require the local Python server.

## Query-To-Evidence Contract

1. A recorded target comes from an archived benchmark packet; a prepared live
   target comes from a reviewed profile; a free-form target is analyst text.
2. Recorded and prepared targets reuse their visible, archived atoms. Edited
   or free-form targets use a deterministic local sentence/clause splitter.
   The demo does not call OpenAI and does not silently paraphrase the query.
3. Eligibility uses original classification and handling metadata, excludes
   `SECRET` and `NOFORN`, enforces strict-before date when supplied, and keeps
   an independent source-ID leakage guard.
4. Document-only retrieval scores every eligible cable against each atom.
   Graph-assisted retrieval adds alias-aware semantic seeds and one-hop graph
   contributions; hierarchy placement explains regions but is not evidence.
5. Verbatim cable segments are extracted from the union of the ranked lists.
   Only citation-aligned passages can certify an atom or bind a redaction fill.
6. The deterministic task layer reports reconstruction coverage, hypothesis
   direction, or shared-candidate constraint coverage. The prose digest only
   explains that result and cannot change it.

Each cross-hierarchy bridge auto-plays once per browser session. Revisiting it
settles directly at the endpoint; the explicit replay control runs the same
ascent/crossing/descent at twice the first-play speed. An exact terminal
segment and its parent candidate cable are consolidated into one destination
step, with the incoming edge and segment provenance retained.
An identical passage bound to multiple atoms is also presented as one citation
card with every atom binding, while the underlying audit rows remain intact.

## Local Preview

From this repository:

```bash
python3 -m http.server 8767 --directory site
```

Open <http://127.0.0.1:8767/>. This tests the same browser-only artifact that
GitHub Pages deploys.

## Scientific Language

- Graph paths explain discovery; raw cited passages establish evidence.
- Route destinations distinguish validated evidence, partial evidence, and
  non-certifying context through explicit atom-to-passage bindings.
- Coverage is atom completeness, not confidence or probability.
- Native deterministic verdicts and separate citation audits remain visibly
  distinct; explanatory prose cannot rewrite either result.
- A leading person candidate remains unresolved until every required clue is
  supported by validated citations.
- SECRET source documents are already class-ineligible. Source-ID exclusion is
  an independent leakage guard.
