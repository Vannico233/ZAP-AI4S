"""Credit for the final accepted done response in the T1 intervention."""
from tube_agent.lab.credit import failed_done


def failed_final_done(result, final_row):
    # termination may be "max_steps" when done itself was the 40th action.
    return (result["error"] is None and result["success"] is False
            and final_row["step"] == result["steps"]
            and failed_done(final_row))


def response_advantage(group_advantage, step, adjusted_step):
    if step == adjusted_step:
        return min(group_advantage, 0.0)
    return group_advantage
