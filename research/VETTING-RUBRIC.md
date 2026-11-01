# Candidate vetting rubric (phase 3)

Every candidate contribution is scored by independent vetters on these axes. A candidate that
fails a HARD gate is out regardless of score.

## HARD gates (pass/fail, with evidence)
G1 Not a duplicate: no open PR in gh/prs_open.json (title/body/files) implements the same thing;
   no merged PR in the last 90 days already shipped it; if an abandoned/stale PR exists, the
   candidate must be materially different and the difference must be stated.
G2 Not reserved: no maintainer has stated (roadmap/RFC comment) that they are doing it themselves
   imminently, and it is not inside an area with an active refactor that would reject outside PRs.
G3 Self-contained: one PR, touching a bounded set of files, no engine (vLLM/SGLang) fork needed,
   no new heavy dependency.
G4 Provable: there is a benchmark (Modal H100/H200) or a CPU regression test that will show the
   claimed effect with numbers, and the metric is one maintainers accept (step wall time, tokens/s,
   peak memory, bitwise-identical outputs for a refactor, a failing test that now passes).
G5 Default-path impact: the change affects what a DEFAULT user gets (or a very common opt-in),
   not an exotic configuration; or, for "novel", it is explicitly requested in issues/RFCs.

## Scores (1-5 each)
S1 Speed / value: expected wall-clock or memory saving for the 8xH100 / Qwen3-8B / GRPO n=8 /
   8k-response reference user, or the size of the unlocked capability.
S2 Demand: issues/RFC reactions, maintainer quotes, competitor precedent.
S3 Certainty: how sure are we the effect is real (measured > derived > guessed).
S4 Merge-ability: fits verl's conventions, small diff, tests, docs; low conflict with in-flight PRs;
   a known reviewer owns the area.
S5 Novelty: would it be the first implementation in verl of something other frameworks or papers
   already proved valuable, or a genuinely new idea with a clear argument.

## Adversarial questions every vetter must answer in writing
A1 What is the strongest reason this will NOT be merged?
A2 What could make the measured gain disappear (a flag already on by default, a newer vLLM version,
   a different model size)?
A3 What correctness risk does it carry, and which existing tests would catch a regression?
A4 Who else is likely working on this right now (name the PR/issue/person)?
A5 If we had to cut scope by half, what is the minimum valuable PR?
