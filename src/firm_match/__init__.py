"""firm_match"""

import re
import string
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from typeguard import typechecked

logger.add(
    Path("logs/logging_for_firmmatch_init.log"), rotation="50 MB"
)  # Automatically rotate too big file
# ---------------------------------------------------------------------------
# Settings and global vars
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
@typechecked
def _char_replace(string: str, substitutions: typing.Dict[str, str]) -> str:
    """Makes a series of text replacements in one go (to be faster).

    Args:
        string (str): String with subsections to be replaced.
        substitutions (typing.Dict[str, str]): Dictionary of replacements.

    Returns:
        str: String with replacements made.
    """
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile("|".join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)


@typechecked
def clean_firm_names(firm_series: pd.Series) -> pd.Series:
    """Carries out a series of operations on firm names, including:
        - All text in lower case
        - Ampersands converted to 'and'
        - Punctuation removal
        - Whitespace removal
        - Removal of some firm-related words, eg plc; also "the"

    Args:
        firm_series (pd.Series): Pandas series with firm names in.

    Raises:
        ValueError: _description_

    Returns:
        pd.Series: Pandas series with cleaned firm names in.

    Examples
    --------
    >>> df = pd.DataFrame(
        {
            "firm_names": [
                "   the   big & little company  __  LimiteD",
            ]
        }
        )
    >>> clean_series = clean_firm_names(df["firm_names"])
    >>> clean_series.iloc[0]
    'big and little company'
    """
    # Everything in lower case
    xf = firm_series.str.lower()
    # Consistent "and"
    sub_text = {"&": "and"}
    xf = xf.map(lambda x: _char_replace(x, sub_text))
    # Replace remaining punctuation with whitespace
    xf = xf.str.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))  # type: ignore
    )
    xf = xf.str.replace(r"\s\s+", " ", regex=True)
    # # Stop words:
    stop_words = ["limited", "the", "ltd", "plc"]
    pat = r"\b(?:{})\b".format("|".join(stop_words))
    xf = xf.str.replace(pat, "", regex=True)
    xf = xf.str.replace(r"\s+", " ", regex=True)
    xf = xf.str.rstrip()
    xf = xf.str.lstrip()
    xf = xf.astype("str")
    return xf


@typechecked
def match_firm_names(
    prime: pd.Series,
    secon: typing.Union[pd.Series, None] = None,
) -> pd.DataFrame:
    """Matching capability for firm names, for one series (deduplication) or two series (matching each entry in prime to its closest equivalent in secon).

    This script provides an arbitrary matching capability. It is asymmetric in the
    sense that one of the input datasets, prime, will be used to define the
    vector space that is used to do the matching.
    It takes in pandas series, prime and secon, matches them, and returns
    an output data frame.

    Args:
        prime (pd.Series): The primary firm name series. Used to create vector space.
        secon (pd.Series, optional): The secondary firm name series (do not provide if deduplicating prime). Defaults to None.

    Returns:
        pd.DataFrame: A list of firms with match scores. Has length same to prime.

    Examples
    --------
    >>> df = pd.DataFrame(
        {
            "firm_names": [
                "the big firm limited",
                "the little firm plc",
                "a little firm",
                "biggest firm",
            ]
        }
        )
    >>> xf = pd.DataFrame(
        {
            "firm_names_secon": [
                "a big firm plc",
                "LitTTle company limited",
                "Biggest Firm",
            ]
        }
        )
    >>> matches = match_firm_names(
        df["firm_names"], xf["firm_names_secon"]
        )
    """

    def clean_names(
        firm_name_series: pd.Series, input_col_name: str, clean_col_name: str
    ) -> pd.DataFrame:
        firm_name_series = firm_name_series.astype("string")
        clean_firm_name_series = clean_firm_names(firm_name_series)
        df = pd.concat([firm_name_series, clean_firm_name_series], axis=1)
        df.columns = pd.Index([input_col_name, clean_col_name])
        df = df.drop_duplicates(subset=clean_col_name, keep="first")
        logger.info(f"Cleaning on {len(df)} input firm names complete")
        return df

    input_col_name = "firm_names"
    clean_col_name = "firm_names_clean"
    prefix_for_secon = "secon_"  # Simply to avoid dataframes with repeated col names
    prime_df = clean_names(prime, input_col_name, clean_col_name)
    secon_flag = True
    if secon is None:
        secon_df = prime_df
        secon_flag = False

        secon_df = secon_df.rename(
            columns=dict(
                zip(
                    [x for x in secon_df.columns],
                    [prefix_for_secon + x for x in secon_df.columns],
                )
            )
        )
    else:
        secon_df = clean_names(
            secon, prefix_for_secon + input_col_name, prefix_for_secon + clean_col_name
        )
        secon_df[prefix_for_secon + clean_col_name] = secon_df[
            prefix_for_secon + clean_col_name
        ].astype("string")
        secon_df = secon_df.dropna(subset=[prefix_for_secon + clean_col_name])

    # Prep for matching non-exacts
    prime_df[clean_col_name] = prime_df[clean_col_name].astype("string")
    prime_df = prime_df.dropna(subset=[clean_col_name])

    # ----------------------
    # Matching
    # ----------------------
    # Method - use char level n-grams
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 4), max_features=30000, encoding="utf-8"
    )
    # Create the tf-idf terms based on characters using the prime
    tfidf_prime = vectorizer.fit_transform(prime_df[clean_col_name])
    tfidf_secon = vectorizer.transform(secon_df[prefix_for_secon + clean_col_name])
    num_matches = 2  # Must be at least 2 as otherwise get only diagonal returned when working in deduplication mode.
    threshold = 0
    mat_of_scores = sp_matmul_topn(
        tfidf_prime, tfidf_secon, threshold=threshold, top_n=num_matches, n_threads=4
    )
    if not secon_flag:
        # If secon and prime are same datasets, fill diagonal with zero as this is
        # the trivial match
        mat_of_scores.setdiag(0)
    max_indexes = np.squeeze(np.asarray(np.argmax(mat_of_scores, 1)))
    top_scores = np.max(mat_of_scores, 1)  # vector of size secon entries
    matches = pd.concat(
        [
            prime_df.reset_index(drop=True),
            secon_df.iloc[max_indexes].reset_index(drop=True),
            pd.DataFrame(top_scores.toarray()),
        ],
        axis=1,
        join="inner",
    )
    matches = matches.reset_index(drop=True).rename(columns={0: "match_score"})
    return matches
