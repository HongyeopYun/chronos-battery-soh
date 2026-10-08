"""Predefined, physical-cell-disjoint folds for small-budget head adaptation."""
import re

from src.battery_fewshot_tuning import check_battery_split

GROUPS = ("nasa_aging", "calce_cs2_1c", "mit_batch1", "mit_batch2", "mit_batch3", "xjtu_2c", "xjtu_3c")
WINDOWS_PER_SOURCE_CELL = 4


def natural_key(value):
    return tuple(int(s) if s.isdigit() else s for s in re.split(r"(\d+)", value))


def check_physical_split(fold, physical):
    check_battery_split(fold["train_ids"], fold["validation_ids"], fold["test_ids"])
    groups = [{physical[b] for b in fold[key]} for key in ("train_ids", "validation_ids", "test_ids")]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("The same physical cell occurs in different splits")


def make_folds(group, records, repeated_physical_ids=()):
    ids = sorted((r["battery_id"] for r in records), key=natural_key)
    physical = {r["battery_id"]: r["physical_battery_id"] for r in records}
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate battery record ID")
    if group in ("nasa_aging", "calce_cs2_1c"):
        if len(ids) != 4:
            raise ValueError("Expected four eligible cells for the predefined four-fold protocol")
        folds = []
        for i, test in enumerate(ids):
            val = ids[(i + 1) % len(ids)]
            folds.append(dict(fold_id=f"test_{test}", train_ids=[b for b in ids if b not in (test, val)],
                              validation_ids=[val], test_ids=[test]))
    elif group.startswith("xjtu_"):
        train = [b for b in ids if b.rsplit("-", 1)[-1] in ("4", "5")]
        val = [b for b in ids if b.rsplit("-", 1)[-1] == "6"]
        folds = [dict(fold_id="fixed", train_ids=train, validation_ids=val,
                      test_ids=[b for b in ids if b not in train + val])]
    elif group.startswith("mit_"):
        # Repeated records are always reserved for test, across all MIT batches.
        sources = [b for b in ids if physical[b] not in repeated_physical_ids]
        if len(sources) < 4:
            raise ValueError("Insufficient eligible unique physical cells")
        train, val = sources[:2], sources[2:3]
        folds = [dict(fold_id="fixed", train_ids=train, validation_ids=val,
                      test_ids=[b for b in ids if physical[b] not in {physical[a] for a in train + val}])]
    else:
        raise ValueError(group)
    for fold in folds:
        if len(fold["train_ids"]) != 2 or len(fold["validation_ids"]) != 1:
            raise ValueError("The budget requires two train and one validation cell")
        check_physical_split(fold, physical)
    all_test = [b for fold in folds for b in fold["test_ids"]]
    if len(all_test) != len(set(all_test)):
        raise ValueError("Each held-out test record must be scored exactly once")
    return folds
