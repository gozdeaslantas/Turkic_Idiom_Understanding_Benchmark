from src.dataset.build_long_df import build_long_dataframe
from src.dataset.split import split_by_group
from src.finetune.tr_only import run_tr_only_ft
from src.finetune.multilingual import run_multilingual_ft
from src.finetune.language_specific import run_language_ft

CSV = "data/raw/multi_turkic_idioms_with_meanings.csv"

df_long = build_long_dataframe(CSV)
df_train, df_test = split_by_group(df_long)

# TR-only
run_tr_only_ft("ytu-ce-cosmos/turkish-e5-large", df_train, "models/ft_tr/e5")

# Multi
run_multilingual_ft("ytu-ce-cosmos/turkish-e5-large", df_train, "models/ft_multi/e5")

# Lang-specific
for L in ["AZ","TK","GA","UZ"]:
    run_language_ft(
        "ytu-ce-cosmos/turkish-e5-large",
        df_train,
        L,
        f"models/ft_lang/e5_{L}"
    )
