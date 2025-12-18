from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_group(
    df_long: pd.DataFrame,
    group_col: str = "group_id",
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train / test by GROUP (idiom group),
    ensuring no group leakage.

    This is a direct functional equivalent of the Colab code:

        GROUPS = df_long["group_id"].unique()
        train_gid, test_gid = train_test_split(...)
        df_train = df_long[df_long["group_id"].isin(train_gid)]
        df_test  = df_long[df_long["group_id"].isin(test_gid)]

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe with `group_id` column.
    group_col : str
        Column name used for grouping.
    test_size : float
        Fraction of groups used for test.
    random_state : int
        RNG seed.
    shuffle : bool
        Whether to shuffle groups before splitting.
    verbose : bool
        Print split statistics.

    Returns
    -------
    df_train : pd.DataFrame
    df_test  : pd.DataFrame
    """

    if group_col not in df_long.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")

    groups = df_long[group_col].unique()

    if verbose:
        print("Total groups:", len(groups))

    train_gid, test_gid = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    df_train = (
        df_long[df_long[group_col].isin(train_gid)]
        .reset_index(drop=True)
    )
    df_test = (
        df_long[df_long[group_col].isin(test_gid)]
        .reset_index(drop=True)
    )

    if verbose:
        print("Train groups:", len(train_gid), "Test groups:", len(test_gid))
        print("Train size:", len(df_train), "Test size:", len(df_test))

    return df_train, df_test
