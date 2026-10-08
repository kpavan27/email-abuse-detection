# Design notes

Why the project is built the way it is, and what I would change at larger scale.

## Problem framing

The deliverable is a *decision* (block or allow), not a probability. So the first
question is which error matters more. Blocking a legitimate email silently loses
something a user needed; letting a spam through costs a second of their attention. The
operating point is therefore framed as a constraint: maximise recall subject to a
false-positive rate of at most 0.5%. It is chosen on validation data and frozen before
test data is touched.

Accuracy and F1 are reported for completeness but not used for selection. On a balanced
corpus they hide exactly the trade-off that matters.

## Data handling

| Decision | Reason | Alternative considered |
|---|---|---|
| Pin the download by SHA-256 | A silently changed upstream file would change every number without anyone noticing | Vendoring the data: rejected, as the repo should not redistribute the corpus |
| De-duplicate before splitting | 13.6% of messages are near-identical copies; otherwise the test set measures memorisation | De-duplicating within splits only: still leaks across splits |
| Group the split by template key | Campaign variants differ only in numbers or whitespace and should stay on one side | MinHash/LSH near-duplicate clustering: better recall of variants, more machinery than this corpus needs |
| Delete identity tokens and years | They separate the classes in this corpus and nowhere else | Placeholder token: tried, failed (the model learned the placeholder) |
| Bundle masking into the artifact | Train/serve skew is a common, silent production failure | Documenting "remember to mask": relies on people |

The identity-token list was not guessed. It was read off the top ham-weighted features
of an unmasked model, and the pipeline re-runs that model on every execution so the
evidence stays current.

## Model choice

Linear models over sparse TF-IDF features:

* They train in about a minute on two CPUs and are easy to calibrate and threshold.
* Their weights can be read. That is how both shortcuts were found; an opaque model with
  the same test score would have hidden them.
* Character n-grams (`char_wb`, 3-5) add robustness to the obfuscation spammers use
  (`v1agra`, inserted symbols) at modest cost.

Hyperparameters: `C` is searched on the validation split with the vectoriser fitted once
and the matrices cached. Validation PR-AUC is flat above C = 32 (differences below
0.0001), so the edge of the grid is not hiding a better optimum worth chasing.

**What I would try next and why it is not here.** A fine-tuned transformer (for example
DistilBERT) would probably close part of the gap on very short messages. It was left
out because the measured failure mode is *too little text*, which a better text model
helps only partly, and because the comparison would need the same leakage controls
applied to sub-word tokens, which is its own piece of work.

## Evaluation

* **Uncertainty:** class-stratified percentile bootstrap (1,000 resamples) at the frozen
  threshold. Stratifying keeps the number of positives stable across resamples.
* **Model comparison:** McNemar's test on paired predictions, because both models score
  the same messages and an unpaired comparison of two accuracies throws that away.
* **Drift:** a temporal hold-out on the spam class only (ham and spam come from different
  years, so a global temporal split would just be a label split), plus the PSI of the
  score distribution as a monitoring metric.
* **Error analysis:** misses are tagged with non-exclusive rules and compared with the
  base rate of each tag in the same class. *Lift*, not raw count, identifies a failure
  mode, since common traits are common in errors too.

## Production considerations not implemented

* Calibration monitoring and threshold re-selection as the spam/ham mix changes.
* A feedback loop from user "not spam" reports into labelled data, with care for the
  selection bias that introduces.
* Non-text features (sender reputation, URL reputation, header anomalies), which the
  error analysis suggests matter most for short messages.
* Shadow deployment of a retrained model, comparing decisions before switching traffic.
