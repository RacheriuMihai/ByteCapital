# app/streamlit_app.py — dashboard-ul Road to Emerging Market
import math

import streamlit as st

from engine import pipeline

st.set_page_config(page_title="Road to Emerging Market", layout="wide")
st.title("Road to Emerging Market")

st.sidebar.header("Scenariu")
levers = {
    "volume": st.sidebar.slider("Volum zilnic (%)", -50, 200, 0) / 100,
    "pension": st.sidebar.slider("Activitate Pilon II (%)", -50, 200, 0) / 100,
    "state_sell": st.sidebar.slider("Vânzare stat (% acțiuni)", 0, 20, 0) / 100,
    "fx": st.sidebar.slider("Curs RON/USD (%)", -15, 15, 0) / 100,
}


@st.cache_data
def run_scenario(levers_tuple):
    return pipeline.run(dict(levers_tuple))


res = run_scenario(tuple(sorted(levers.items())))

c1, c2, c3 = st.columns(3)
c1.metric("Companii care trec toate testele", res.n_pass_all,
          f"min. {int(res.th.min_companies)}", delta_color="off")
c2.metric("Evaluări consecutive", res.country_streak,
          f"din {int(res.th.reviews_needed)}", delta_color="off")
c3.metric("Curs BNR folosit", "—" if math.isnan(res.fx) else f"{res.fx:.4f}")

if res.tests.empty:
    st.info("Nu există încă date. Completați data/trades.csv, data/shareholders.csv, "
            "data/issuers.csv și rulați `python ingest/bnr.py` pentru curs.")
else:
    st.subheader("Teste pe companie (ultima lună)")
    st.dataframe(res.tests, width="stretch")

    st.subheader("Lista de deficit de lichiditate")
    st.dataframe(res.gap_list, width="stretch")
    st.download_button("Export CSV", res.gap_list.to_csv(index=False), "gap_list.csv")
