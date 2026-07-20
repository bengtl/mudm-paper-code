from mudm_paper.leakage_analysis import serial_homolog_exposure


def _toy():
    """One task class X, spread over 7 groups and 2 cell types; a bilateral
    pair (A_L/A_R) shares group 'A'."""
    body_ids = ["A_L", "A_R", "B", "C", "D", "E", "F", "G"]
    task_labels = {b: "X" for b in body_ids}
    cell_type = {"A_L": "t1", "A_R": "t1", "B": "t1", "C": "t1",
                 "D": "t2", "E": "t2", "F": "t2", "G": "t2"}
    group_of = {"A_L": "A", "A_R": "A", "B": "B", "C": "C",
                "D": "D", "E": "E", "F": "F", "G": "G"}
    return body_ids, task_labels, cell_type, group_of


def test_no_group_straddle_and_structure():
    body_ids, task_labels, cell_type, group_of = _toy()
    r = serial_homolog_exposure(body_ids, task_labels, cell_type, group_of,
                                seeds=5, min_class=2)
    # Groups never cross the split (the bilateral-leakage guarantee).
    assert r["straddle"] == 0.0
    # Single task class -> its class is always represented in train.
    assert r["same_class"] == 1.0
    # Structure: 7 split groups, 2 cell types, exposure is a valid fraction.
    assert r["n_groups"] == 7 and r["n_cell_types"] == 2
    assert 0.0 <= r["serial"] <= 1.0


def test_no_serial_exposure_when_cell_type_equals_group():
    # If each cell type lives in exactly one group, no serial homolog can be
    # in a different group -> serial exposure is 0.
    body_ids, task_labels, _ct, group_of = _toy()
    cell_type = {b: group_of[b] for b in body_ids}  # one cell type per group
    r = serial_homolog_exposure(body_ids, task_labels, cell_type, group_of,
                                seeds=5, min_class=2)
    assert r["serial"] == 0.0
    assert r["straddle"] == 0.0
