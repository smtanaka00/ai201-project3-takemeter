# outputs/

Artifacts produced by running `notebooks/takemeter_colab.ipynb` in Colab. Commit these —
they back the evaluation tables and error analysis in the top-level `README.md`.

| File | Produced by | Used for |
|------|-------------|----------|
| `evaluation_results.json` | Section 6 | Headline accuracy comparison (baseline vs fine-tuned). |
| `confusion_matrix.png` | Section 4 | Fine-tuned model's per-class confusion (shown in README). |
| `test_predictions.csv` | Section 6 | Per-row predictions → input to `src/error_analysis.py`. |
