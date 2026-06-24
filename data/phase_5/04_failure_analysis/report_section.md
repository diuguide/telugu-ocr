# Failure analysis

**Failure rule:** For each processed model: page CER >= 0.80 AND (GPT-4o score <= 2 OR missing-output rate >= 0.30). A shared failure satisfies the rule for both models.

Shared failures: 3 of 40 pages.

| association_rank | feature | failure_mean | nonfailure_mean | standardized_mean_difference |
|---|---|---|---|---|
| 1 | mean_intensity | 214.4761 | 213.6891 | 0.7058 |
| 2 | foreground_density | 0.0539 | 0.0569 | -0.4148 |
| 3 | diacritic_density | 0.3860 | 0.3902 | -0.2774 |
| 4 | contrast_std | 23.1280 | 22.4869 | 0.2773 |
| 5 | conjunct_density | 0.0743 | 0.0767 | -0.2155 |
| 6 | height | 292.8628 | 292.6445 | 0.1369 |
| 7 | reference_length | 227.6667 | 229.3784 | -0.1300 |
| 8 | width | 1224.9489 | 1218.2887 | 0.0885 |

These associations describe the evaluated sample; they are not causal estimates. Telugu conjunct and vowel-sign densities are reported explicitly to connect failures to script structure.
