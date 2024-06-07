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
        str.maketrans(string.punctuation, " " * len(string.punctuation))
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
    prime: pd.DataFrame,
    secon: typing.Union[pd.DataFrame, None] = None,
    p_match_col: str = "prime_name",
    s_match_col: str = "secon_name",
) -> pd.DataFrame:
    """Matching capability for firm names, for one dataframe (deduplication) or two dataframes (matching each entry in prime to its closest equivalent in secon).

    This script provides an arbitrary matching capability. It is asymmetric in the
    sense that one of the input datasets, prime, will be used to define the
    vector space that is used to do the matching.
    It takes in pandas dataframes, prime and secon, matches them, and returns
    an output data frame.

    Args:
        prime (pd.DataFrame): The primary firm name dataframe. Used to create vector space.
        secon (pd.DataFrame, optional): The secondary firm name dataframe, do not provide if deduplicating prime. Defaults to None.
        p_match_col (str, optional): The name of the column containing firm names in prime. Defaults to "prime_name".
        s_match_col (str, optional):  The name of the column containing firm names in secon. Defaults to "secon_name".

    Returns:
        pd.DataFrame: A list of firms with match scores.

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
        df, xf, p_match_col="firm_names", s_match_col="firm_names_secon"
        )
    """

    def clean_names(df: pd.DataFrame, col: str) -> pd.DataFrame:
        df.loc[:, col] = df[col].astype(str)
        df.loc[:, col + "_cln"] = clean_firm_names(df[col])
        df = df.drop_duplicates(subset=col + "_cln", keep="first")
        logger.info("Cleaning on " + col + f" complete; {len(df)} records")
        return df

    if type(prime) != pd.DataFrame:
        raise ValueError("prime should be a dataframe.")

    prime = clean_names(prime, p_match_col)
    secon_flag = True
    if secon is None:
        secon = prime
        s_match_col = p_match_col
        secon_flag = False
        prefix = "secon_"
        secon = secon.rename(
            columns=dict(
                zip([x for x in secon.columns], [prefix + x for x in secon.columns])
            )
        )
        s_match_col = prefix + p_match_col
    else:
        if type(secon) != pd.DataFrame:
            raise ValueError("secon should be a dataframe.")
        secon = clean_names(secon, s_match_col)
        secon.loc[:, s_match_col + "_cln"] = secon[s_match_col + "_cln"].astype(
            "string"
        )
        secon = secon.dropna(subset=[s_match_col + "_cln"])

    # Prep for matching non-exacts
    prime.loc[:, p_match_col + "_cln"] = prime[p_match_col + "_cln"].astype("string")
    prime = prime.dropna(subset=[p_match_col + "_cln"])

    # ----------------------
    # Matching
    # ----------------------
    # Method - use char level n-grams
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 4), max_features=30000, encoding="utf-8"
    )
    # Create the tf-idf terms based on characters using the prime
    tfidf_prime = vectorizer.fit_transform(prime[p_match_col + "_cln"])
    tfidf_secon = vectorizer.transform(secon[s_match_col + "_cln"])
    num_matches = 2
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
            prime.reset_index(drop=True),
            secon.iloc[max_indexes].reset_index(drop=True),
            pd.DataFrame(top_scores.toarray()),
        ],
        axis=1,
        join="inner",
    )
    matches = matches.reset_index(drop=True).rename(columns={0: "match_score"})
    return matches
