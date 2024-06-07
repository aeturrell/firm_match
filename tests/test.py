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
                "big ltd",
            ]
        }
    )
    matches_scores_answer = np.array([0.778495, 0.932603, 0.932603, 0.778495])
    matches = match_firm_names(df["firm_names"])
    assert (
        matches_scores_answer.round(2) == matches["match_score"].values.round(2)
    ).all()


def test_matching_from_two_lists_of_firm_data():
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
    xf = pd.DataFrame(
        {
            "firm_names_secon": [
                "a big firm plc",
                "LitTTle company limited",
                "Biggest Firm",
            ]
        }
    )
    matches = match_firm_names(
        df["firm_names"], xf["firm_names_secon"]
    )
    matches_scores_answer = np.array([0.906213, 0.865682, 0.846631, 1.000000])
    assert (
        matches_scores_answer.round(2) == matches["match_score"].values.round(2)
    ).all()
