# Error analysis

| configuration | category | affected_words | word_rate |
|---|---|---|---|
| surya/raw | base_character_substitution | 190 | 0.3711 |
| surya/raw | deletion | 327 | 0.6387 |
| surya/raw | diacritic_or_vowel_sign_error | 465 | 0.9082 |
| surya/raw | hallucinated_extra_text | 22 | 0.0430 |
| surya/raw | insertion | 114 | 0.2227 |
| surya/raw | missing_output | 256 | 0.5000 |
| surya/raw | punctuation_or_digit_error | 0 | 0.0000 |
| surya/raw | virama_or_conjunct_error | 247 | 0.4824 |
| surya/raw | word_boundary_split_or_join | 123 | 0.2402 |
| surya/processed | base_character_substitution | 355 | 0.3694 |
| surya/processed | deletion | 592 | 0.6160 |
| surya/processed | diacritic_or_vowel_sign_error | 861 | 0.8959 |
| surya/processed | hallucinated_extra_text | 75 | 0.0780 |
| surya/processed | insertion | 253 | 0.2633 |
| surya/processed | missing_output | 483 | 0.5026 |
| surya/processed | punctuation_or_digit_error | 2 | 0.0021 |
| surya/processed | virama_or_conjunct_error | 499 | 0.5193 |
| surya/processed | word_boundary_split_or_join | 220 | 0.2289 |
| tesseract/raw | base_character_substitution | 160 | 0.1665 |
| tesseract/raw | deletion | 833 | 0.8668 |
| tesseract/raw | diacritic_or_vowel_sign_error | 912 | 0.9490 |
| tesseract/raw | hallucinated_extra_text | 9 | 0.0094 |
| tesseract/raw | insertion | 79 | 0.0822 |
| tesseract/raw | missing_output | 730 | 0.7596 |
| tesseract/raw | punctuation_or_digit_error | 2 | 0.0021 |
| tesseract/raw | virama_or_conjunct_error | 525 | 0.5463 |
| tesseract/raw | word_boundary_split_or_join | 36 | 0.0375 |
| tesseract/processed | base_character_substitution | 507 | 0.5276 |
| tesseract/processed | deletion | 637 | 0.6629 |
| tesseract/processed | diacritic_or_vowel_sign_error | 894 | 0.9303 |
| tesseract/processed | hallucinated_extra_text | 41 | 0.0427 |
| tesseract/processed | insertion | 222 | 0.2310 |
| tesseract/processed | missing_output | 280 | 0.2914 |
| tesseract/processed | punctuation_or_digit_error | 20 | 0.0208 |
| tesseract/processed | virama_or_conjunct_error | 509 | 0.5297 |
| tesseract/processed | word_boundary_split_or_join | 301 | 0.3132 |

Categories are assigned from normalized Unicode alignments. GPT-4o detections are retained as a separate evidence source, avoiding false precision from treating LLM judgments as ground truth. Conjunct and diacritic subset results are reported separately.
