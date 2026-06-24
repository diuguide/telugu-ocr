# Model comparison

| accuracy_rank | configuration | corpus_cer | corpus_wer | coverage | mean_llm_score | mean_flagged_region_rate |
|---|---|---|---|---|---|---|
| 1 | surya/raw | 0.6773 | 292.0000 | 0.5328 | 2.0455 | 0.4567 |
| 2 | tesseract/processed | 0.6874 | 537.0000 | 1.0000 | 1.7500 | 0.6186 |
| 3 | surya/processed | 0.6941 | 568.0000 | 1.0000 | 1.9750 | 0.4380 |
| 4 | tesseract/raw | 0.7961 | 200.0000 | 1.0000 | 1.2250 | 0.5648 |

The strictly paired comparison contains 512 word regions. Overall and paired results are kept separate because Surya raw coverage is incomplete. CER, WER, and LLM lexical-quality scores measure different behavior and are not treated as interchangeable.
