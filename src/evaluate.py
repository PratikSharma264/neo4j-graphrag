import argparse
import json
import re
from pathlib import Path

from src.query import answer_question, driver, retrieve_chunks

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT_DIR / "evaluation" / "questions.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "evaluation" / "results.json"

CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
SOURCE_PATTERN = re.compile(r"^\[(S\d+)\]\s+(.+),\s+page\s+\d+$")


def normalize_source(source: str) -> str:
    return source.replace("\\", "/").casefold()


def load_cases(dataset_path: Path) -> list[dict]:
    cases = []

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            case = json.loads(line)
            required = {
                "id",
                "type",
                "question",
                "answerable",
                "relevant_sources",
                "reference_answer",
            }
            missing = required.difference(case)
            if missing:
                raise ValueError(
                    f"{dataset_path}:{line_number}: missing fields: "
                    f"{', '.join(sorted(missing))}"
                )

            cases.append(case)

    if not cases:
        raise ValueError(f"No evaluation cases found in {dataset_path}")

    return cases


def cited_sources(answer: str) -> tuple[list[str], int]:
    answer_text, _, source_text = answer.partition("\n\nRetrieved sources:\n")
    source_map = {}

    for line in source_text.splitlines():
        match = SOURCE_PATTERN.match(line.strip())
        if match:
            source_map[match.group(1)] = normalize_source(match.group(2))

    citation_ids = list(dict.fromkeys(CITATION_PATTERN.findall(answer_text)))
    paths = [source_map[citation] for citation in citation_ids if citation in source_map]
    invalid_count = sum(citation not in source_map for citation in citation_ids)
    return paths, invalid_count


def citation_metrics(answer: str, expected_sources: set[str]) -> dict:
    paths, invalid_count = cited_sources(answer)
    cited = set(paths)

    if not expected_sources:
        return {
            "precision": 1.0 if not paths and not invalid_count else 0.0,
            "recall": None,
        }

    citation_count = len(paths) + invalid_count
    precision = (
        sum(path in expected_sources for path in paths) / citation_count
        if citation_count
        else 0.0
    )
    recall = len(cited & expected_sources) / len(expected_sources)

    return {"precision": round(precision, 4), "recall": round(recall, 4)}


def retrieval_metrics(chunks: list[dict], expected_sources: set[str]) -> dict:
    ranked_sources = list(
        dict.fromkeys(
            normalize_source(chunk["source_path"])
            for chunk in chunks
            if chunk.get("source_path")
        )
    )

    if not expected_sources:
        return {"ranked_sources": ranked_sources, "hit": None, "reciprocal_rank": None}

    rank = next(
        (
            index
            for index, source in enumerate(ranked_sources, start=1)
            if source in expected_sources
        ),
        None,
    )

    return {
        "ranked_sources": ranked_sources,
        "hit": rank is not None,
        "reciprocal_rank": round(1 / rank, 4) if rank else 0.0,
    }


def average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def evaluate_case(case: dict) -> dict:
    chunks = retrieve_chunks(case["question"])
    expected_sources = {
        normalize_source(source) for source in case["relevant_sources"]
    }

    graph_answer = answer_question(
        case["question"], chunks=chunks, include_graph=True
    )
    vector_answer = answer_question(
        case["question"], chunks=chunks, include_graph=False
    )

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "answerable": case["answerable"],
        "reference_answer": case["reference_answer"],
        "retrieval": retrieval_metrics(chunks, expected_sources),
        "graph_answer": {
            "text": graph_answer,
            "citation_metrics": citation_metrics(graph_answer, expected_sources),
        },
        "vector_only_answer": {
            "text": vector_answer,
            "citation_metrics": citation_metrics(vector_answer, expected_sources),
        },
        "manual_review": {
            "graph_correct": None,
            "graph_faithful": None,
            "graph_abstained_when_unanswerable": None,
            "vector_only_correct": None,
            "vector_only_faithful": None,
            "vector_only_abstained_when_unanswerable": None,
        },
    }


def summarize(results: list[dict]) -> dict:
    answerable = [result for result in results if result["answerable"]]

    return {
        "case_count": len(results),
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(results) - len(answerable),
        "retrieval_hit_rate": average(
            [float(result["retrieval"]["hit"]) for result in answerable]
        ),
        "retrieval_mrr": average(
            [result["retrieval"]["reciprocal_rank"] for result in answerable]
        ),
        "graph_citation_precision": average(
            [result["graph_answer"]["citation_metrics"]["precision"] for result in answerable]
        ),
        "graph_citation_recall": average(
            [result["graph_answer"]["citation_metrics"]["recall"] for result in answerable]
        ),
        "vector_only_citation_precision": average(
            [result["vector_only_answer"]["citation_metrics"]["precision"] for result in answerable]
        ),
        "vector_only_citation_recall": average(
            [result["vector_only_answer"]["citation_metrics"]["recall"] for result in answerable]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GraphRAG answers against a labeled question set."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = load_cases(args.dataset)

    try:
        driver.verify_connectivity()
        results = [evaluate_case(case) for case in cases]
    finally:
        driver.close()

    report = {"summary": summarize(results), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], indent=2))
    print(f"Detailed answers and review fields: {args.output}")


if __name__ == "__main__":
    main()