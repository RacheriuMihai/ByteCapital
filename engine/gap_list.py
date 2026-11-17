# engine/gap_list.py — lista de deficit: ce îi lipsește fiecărei companii ca să treacă
import pandas as pd

COLS = [
    "symbol", "tests_passed", "pass_full", "pass_ff", "pass_atvr_12m", "pass_atvr_3m", "pass_fot",
    "atvr_12m", "atvr_3m", "fot_min_q",
    "avg_median_value_ron", "needed_median_value_ron", "value_gap_ron_per_day",
    "full_mcap_usd_m", "ff_mcap_usd_m", "ff_needed_usd_m", "ff_gap_usd_m", "full_gap_usd_m", "fif",
]


def build(latest: pd.DataFrame) -> pd.DataFrame:
    """Companiile care trec cel puțin un test, dar nu pe toate, ordonate de la cea mai
    aproape de pragul complet: întâi după testele trecute, apoi după deficitul zilnic.

    value_gap_ron_per_day: cât trebuie să crească valoarea mediană zilnică (RON) ca ATVR 12L
    să atingă pragul, la aceleași zile tranzacționate și același free float.
    ff_gap_usd_m: cât free float lipsește (mil. USD) pentru testul de capitalizare free float."""
    if latest.empty:
        return pd.DataFrame(columns=COLS)
    g = latest[(latest["tests_passed"] >= 1) & (~latest["pass_all"])]
    return (g[COLS].sort_values(["tests_passed", "value_gap_ron_per_day"], ascending=[False, True])
            .reset_index(drop=True))
