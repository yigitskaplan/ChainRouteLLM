import pandas as pd
import pytest

from src.routing.run_stage2 import measure


def test_quality_and_cost_are_shares_of_always_using_the_reference_model():
    table = pd.DataFrame(
        {
            "sample_id": ["q1", "q2"],
            "big|quality": [1.0, 1.0],
            "big|cost": [3.0, 3.0],
            "small|quality": [1.0, 0.0],
            "small|cost": [1.0, 1.0],
        }
    )
    # The picks list the questions in a different order than the table.
    picks = pd.DataFrame({"sample_id": ["q2", "q1"], "w_C=0": ["big", "big"], "w_C=1": ["small", "big"]})
    shares = measure(table, picks, ["big", "small"], reference="big")

    assert shares["w_C"].tolist() == [0.0, 1.0]
    # w_C=1: q2 goes to small (wrong, cost 1) and q1 to big (right, cost 3).
    assert shares["quality_vs_gpt4"].tolist() == pytest.approx([1.0, 0.5])
    assert shares["cost_vs_gpt4"].tolist() == pytest.approx([1.0, 4 / 6])
