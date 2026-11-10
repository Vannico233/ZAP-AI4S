"""Training-only turn credit; never included in policy observations."""


def failed_done(row):
    return (row.get("action", {}).get("action") == "done"
            and row.get("feedback", {}).get("accepted") is True
            and row["judge_after"]["success"] is False)


def turn_advantage(row, advantage, mode="legacy"):
    if mode not in ("legacy", "nonpositive"):
        raise ValueError("Unknown failed-done credit mode")
    if row["feedback"].get("accepted") is False:
        return -1.
    if mode == "nonpositive" and failed_done(row):
        return min(float(advantage), 0.)
    return float(advantage)


def credit_metrics(trajectories, advantages, mode):
    failed = positive_before = positive_after = 0
    for trajectory, advantage in zip(trajectories, advantages):
        for row in trajectory:
            if failed_done(row):
                failed += 1
                positive_before += int(advantage > 0)
                positive_after += int(turn_advantage(row, advantage, mode) > 0)
    return dict(failed_done_turns=failed, positive_failed_done_before=positive_before,
                positive_failed_done_after=positive_after)
