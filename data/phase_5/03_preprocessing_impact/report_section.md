# Preprocessing impact

| model | comparison | metric | pairs | baseline_mean | candidate_mean | absolute_change | bootstrap_ci_low | bootstrap_ci_high | wins | ties | losses | p_value |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| surya | raw_to_run_1 | cer | 21 | 0.6886 | 0.7227 | 0.0341 | 0.0139 | 0.0539 | 4 | 2 | 15 | 0.0048 |
| surya | raw_to_run_1 | wer | 21 | 1.0221 | 1.0919 | 0.0698 | 0.0031 | 0.1460 | 2 | 10 | 9 | 0.0503 |
| surya | run_1_to_run_2 | cer | 21 | 0.7227 | 0.7176 | -0.0051 | -0.0214 | 0.0117 | 10 | 0 | 11 | 0.5430 |
| surya | run_1_to_run_2 | wer | 21 | 1.0919 | 1.0489 | -0.0430 | -0.1207 | 0.0293 | 9 | 8 | 4 | 0.2213 |
| surya | raw_to_run_2 | cer | 22 | 0.6944 | 0.7201 | 0.0257 | 0.0041 | 0.0479 | 5 | 2 | 15 | 0.0438 |
| surya | raw_to_run_2 | wer | 22 | 1.0393 | 1.0467 | 0.0074 | -0.0566 | 0.0726 | 5 | 11 | 6 | 0.7555 |
| tesseract | raw_to_run_1 | cer | 40 | 0.8538 | 0.7795 | -0.0743 | -0.1061 | -0.0447 | 32 | 0 | 8 | 0.0000 |
| tesseract | raw_to_run_1 | wer | 40 | 0.9905 | 1.2465 | 0.2560 | 0.1574 | 0.3615 | 0 | 18 | 22 | 0.0000 |
| tesseract | run_1_to_run_2 | cer | 40 | 0.7795 | 0.6974 | -0.0820 | -0.0925 | -0.0707 | 39 | 0 | 1 | 0.0000 |
| tesseract | run_1_to_run_2 | wer | 40 | 1.2465 | 1.2666 | 0.0202 | -0.0312 | 0.0700 | 10 | 16 | 14 | 0.3384 |
| tesseract | raw_to_run_2 | cer | 40 | 0.8538 | 0.6974 | -0.1564 | -0.1897 | -0.1254 | 40 | 0 | 0 | 0.0000 |
| tesseract | raw_to_run_2 | wer | 40 | 0.9905 | 1.2666 | 0.2761 | 0.1909 | 0.3611 | 1 | 14 | 25 | 0.0000 |

The analysis reports raw→run 1, run 1→run 2, and raw→run 2 on separate identity-paired cohorts. Negative CER/WER changes indicate improvement. Overall stage summaries are reported separately because Surya coverage differs between stages. LLM validation was not run for run 1 and is therefore shown only for raw→run 2.
