# Manual repetition labels

These annotations are manual reference labels for evaluating tests and analysis results. They are not model-correction instructions or training data.

`manual-repetition-labels.csv` contains one row per repetition:

- `video_id` and `video_filename` identify the normalized local source video.
- `rep_number` is the 1-based repetition number from the manual annotation.
- `range_label` is `complete`, `partial`, or `no_extension`.
- `technique_label` is `perfect`, `perfect_with_pause`, `stop_in_middle`, or `no`.
- `asymmetry_label` is `present` only when explicitly annotated, otherwise `not_recorded`. `not_recorded` does not mean asymmetry was absent.

`perfect_with_pause` preserves the source label “perfect (stop)”. `stop_in_middle` means a pause may have occurred in the middle of the repetition; according to the annotator, it may be mild rather than a pronounced stop. Asymmetry is recorded independently from technique and pause labels.

Only the final, uncrossed annotation is recorded. For example, Pablo 01 repetition 5 is `complete` (the final “Full” label), and Pablo 02 repetition 8 is normalized from the typo “complet” to `complete`.
