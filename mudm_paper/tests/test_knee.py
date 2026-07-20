from mudm_paper.knee import task_appropriate_level, moving_knee

def test_picks_cheapest_within_epsilon():
    levels = [{"level":0,"acc_mean":0.80,"cost":1},{"level":1,"acc_mean":0.90,"cost":5},{"level":2,"acc_mean":0.91,"cost":50}]
    assert task_appropriate_level(levels, ref_acc=0.91, epsilon=0.02) == 1

def test_moving_knee_detects_shift():
    per = {"coarse_task":[{"level":0,"acc_mean":0.9,"cost":1},{"level":2,"acc_mean":0.9,"cost":50}],
           "fine_task":  [{"level":0,"acc_mean":0.6,"cost":1},{"level":2,"acc_mean":0.9,"cost":50}]}
    out = moving_knee(per, ref_by_task={"coarse_task":0.9,"fine_task":0.9}, epsilon=0.02)
    assert out["moves"] is True and out["knees"]["coarse_task"] == 0 and out["knees"]["fine_task"] == 2

def test_no_move_when_same_level():
    per = {"a":[{"level":0,"acc_mean":0.9,"cost":1},{"level":2,"acc_mean":0.9,"cost":50}],
           "b":[{"level":0,"acc_mean":0.9,"cost":1},{"level":2,"acc_mean":0.9,"cost":50}]}
    assert moving_knee(per, {"a":0.9,"b":0.9}, 0.02)["moves"] is False
