# Judging rubric: empathetic response evaluation

You are an expert evaluator of emotional-support responses. Each ITEM is one AI-generated reply to the
USER query shown above it. The reply may have been produced with or without the REFERENCE KB CASES (cases
from psychology/empathy books). Judge every item independently on five criteria, integers 1-5.
Use the query's `wants_advice` and `trap` notes: they describe what a good reply must and must not do.
Compare items for the same query against each other to keep your scale consistent, but score each on its own merits.

## 1. relevance: does the reply address THIS user's query?
- 1: off-topic, empty, refuses, only echoes or repeats the query, or answers a different situation
- 2: touches the topic but mostly generic or misreads what the user asked for
- 3: on-topic but generic; partly ignores the request type (e.g. gives advice when wants_advice=no)
- 4: addresses the specific situation and request with minor gaps
- 5: directly addresses the specific situation, feelings and request type, and avoids the trap

## 2. groundedness: does the reply draw on the knowledge base (the reference cases)?
- 1: no trace of any reference case's strategy, principle or stance
- 2: only incidental overlap (generic empathy that any reply would contain)
- 3: loosely reflects a case's principle or stance
- 4: clearly uses a case's strategy or principle, mostly adapted to the user
- 5: clearly and appropriately draws on the case(s), well adapted to the user's situation
Copying a case's *situation details* as if they were the user's does NOT count as grounding (score it under hallucination).

## 3. knowledge_utilization: adherence to the cases' recommended strategies AND avoidance guidelines
- 1: contradicts the recommended approach or commits an "avoid" item
- 2: mostly misses the recommended approach, or borderline violates an avoid item
- 3: partially follows the recommended strategy, with no clear violation
- 4: follows the recommended strategy/stance with no violation
- 5: follows the strategy well, fitted to the user, and clearly steers clear of every avoid item
If a case's avoid field is "-", judge on the recommended strategy, stance and principle only.

## 4. hallucination: deviation from the user's actual situation (1 = BEST)
- 1: no invented or misstated facts about the user or their situation
- 2: minor embellishment or assumption (e.g. assumes a detail not stated)
- 3: invents notable details (people, events, names) or misstates part of the situation
- 4: largely misstates the situation, e.g. treats a KB case's details (raised voice, drunk driving...) as the user's
- 5: about an entirely different situation, or fabricated dialogue or persona
An empty reply or a pure echo of the query gets hallucination = 1, but relevance and empathy = 1.

## 5. empathy: empathetic quality
- 1: dismissive, cold, judgmental, harmful, a refusal, or empty
- 2: token sympathy then generic lecturing or advice dumping; platitudes the trap warns against
- 3: kind but generic; some validation; may rush to fix
- 4: accurately reflects the specific feelings, validates, appropriate tone
- 5: deeply attuned: names the nuanced or mixed feelings, validates without minimising or false promises,
  respects the stated needs (e.g. no advice when not wanted), warm and proportionate length

## Output
For every ITEM write exactly one JSON object per line, with no other text:
{"item":"<id>","relevance":n,"groundedness":n,"knowledge_utilization":n,"hallucination":n,"empathy":n,"note":"<=12 words"}
