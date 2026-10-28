"""Pairwise metrics against an independently maintained manifest. Labels never enter detection."""

from __future__ import annotations

from itertools import combinations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate_pairwise(documents, reports, manifest) -> dict:
    """Positive pairs are only those listed in the manifest.

    Paraphrase pairs are positives. Shared boilerplate/template pairs listed as
    negative_benign_pairs are treated as negatives even if lexical overlap exists.
    Cross-modal pairs are positives only when included in positive_pairs or
    cross_modal_pairs. Abstentions (failed/empty documents) stay in the pair set
    and count as predicted-negative unless a match was actually reported.
    """
    rows = {item["document_id"]: item for item in manifest["documents"]}
    positives = {"|".join(sorted(pair)) for pair in manifest.get("positive_pairs", [])}
    positives |= {"|".join(sorted(pair)) for pair in manifest.get("cross_modal_pairs", [])}
    actual: list[int] = []
    predicted: list[int] = []
    categories: dict[str, list[int]] = {}
    skipped = []

    for left, right in combinations(sorted(rows), 2):
        key = "|".join(sorted((left, right)))
        positive = key in positives
        detected = False
        for source in (
            reports.get(left, {}).get("peer_matches", []),
            reports.get(right, {}).get("peer_matches", []),
        ):
            if any(
                match.get("source_document_id") in {left, right}
                and match.get("query_document_id") in {left, right}
                and not match.get("boilerplate_discounted")
            for match in source
            ):
                detected = True
        actual.append(int(positive))
        predicted.append(int(detected))
        label = rows[left].get("category", "other")
        slot = categories.setdefault(label, [0, 0])
        slot[0] += int(positive)
        slot[1] += int(detected)
        if reports.get(left, {}).get("status") in {"failed", "empty", "missing"}:
            skipped.append(left)

    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel().tolist()
    return {
        "pairs_evaluated": len(actual),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "per_category_counts": categories,
        "label_policy": {
            "positive_pair": "Manifest positive_pairs and cross_modal_pairs",
            "negative_pair": "All other unordered pairs, including negative_benign_pairs",
            "paraphrase": "Treated as positive when listed in positive_pairs",
            "boilerplate": "Shared prompts are not positives unless explicitly listed",
            "abstention": "Failed/empty documents remain predicted-negative unless a match was written",
        },
        "documents_with_extraction_issues": sorted(set(skipped)),
    }
