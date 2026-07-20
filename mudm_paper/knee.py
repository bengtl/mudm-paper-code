def task_appropriate_level(levels, ref_acc, epsilon):
    """levels: list of {"level": int, "acc_mean": float, "cost": float}.
    Return the CHEAPEST level whose acc_mean >= ref_acc - epsilon;
    if none qualify, return the level with the highest acc_mean."""
    ok = [l for l in levels if l["acc_mean"] >= ref_acc - epsilon]
    if ok:
        return min(ok, key=lambda l: l["cost"])["level"]
    return max(levels, key=lambda l: l["acc_mean"])["level"]


def moving_knee(per_task, ref_by_task, epsilon):
    """per_task: {task: levels-list}; ref_by_task: {task: ref_acc}.
    Returns {"knees": {task: chosen_level}, "moves": bool} where moves is
    True iff the chosen level differs across >= 2 tasks."""
    knees = {t: task_appropriate_level(lv, ref_by_task[t], epsilon)
             for t, lv in per_task.items()}
    return {"knees": knees, "moves": len(set(knees.values())) >= 2}
