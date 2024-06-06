"""Test cases for the __main__ module."""

import numpy as np
import pandas as pd
from firm_match import match_firm_names


def test_self_matching_firm_data():
    df = pd.DataFrame(
        {
            "firm_names": [
                "the big firm limited",
                "the little firm plc",
                "a little firm",
                "biggest firm",
            ]
        }
    )
    matches_scores_answer = np.array([0.62759711, 0.9266666, 0.9266666, 0.62759711])
    matches = match_firm_names(df, p_match_col="firm_names")
    assert (
        matches_scores_answer.round(2) == matches["match_score"].values.round(2)
    ).all()
