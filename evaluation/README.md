# GraphRAG Evaluation

Run from the repository root after Neo4j and Ollama are available and the knowledge base is indexed:

```powershell
python -m src.evaluate
```

The benchmark in `questions.jsonl` includes direct, relationship-based, and unanswerable questions. Each answerable question lists the source document(s) that should support its answer.

The report compares GraphRAG answers with vector-only answers using the same retrieved chunks. It reports source-level retrieval hit rate and mean reciprocal rank, plus citation precision and recall. These metrics measure source matching, not clinical correctness. Review the generated answers against `reference_answer` and the cited evidence, then fill in the `manual_review` fields for correctness, faithfulness, and abstention on unanswerable cases.

Use `--dataset` and `--output` to select different files. Expand the benchmark with manually verified cases before drawing conclusions; the included set is a smoke-test baseline, not a statistically representative evaluation.