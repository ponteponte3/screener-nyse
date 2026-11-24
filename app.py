import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="AlphaPulse Screener NYSE",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaPulse — Screener Tecnico & Fondamentale NYSE (EUR)")
st.markdown("1. **Filtro Tecnico:** `Prezzo > EMA9`, `RSI`, `Volume` + Position Sizing in Euro.\n2. **Filtro Fondamentale:** Analisi su richiesta dei candidati (Multipli, Crescita, Debito, Trimestrali).")

# --- RECUPERO CAMBIO EUR/USD LIVE ---
@st.cache_data(ttl=3600)
def get_cambio_live():
    try:
        data = yf.download("EURUSD=X", period="5d", interval="1d", progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return round(float(data['Close'].iloc[-1]), 4)
    except Exception:
        pass
    return 1.08

cambio_default = get_cambio_live()

# --- BARRA LATERALE (PARAMETRI INTERATTIVI) ---
st.sidebar.header("⚙️ Parametri Operativi")
capitale_eur = st.sidebar.number_input("Capitale Totale (€)", min_value=500.0, value=10000.0, step=500.0)
rischio_max_pct = st.sidebar.slider("Rischio Max per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5) / 100
allocazione_max_pct = st.sidebar.slider("Allocazione Max per Trade (%)", min_value=5.0, max_value=100.0, value=20.0, step=5.0) / 100
rapporto_rr = st.sidebar.slider("Rapporto Rischio/Rendimento (R:R)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
moltiplicatore_atr = st.sidebar.slider("Moltiplicatore ATR (Stop Loss)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🔍 Filtri Tecnici & Liquidità")
rsi_range = st.sidebar.slider("Range RSI (14g)", min_value=30, max_value=80, value=(50, 68))
rel_vol_min = st.sidebar.slider("Volume Relativo Minimo (vs Media 30g)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
prezzo_minimo_usd = st.sidebar.number_input("Prezzo Minimo Titolo ($)", min_value=1.0, value=5.0, step=1.0)
volume_minimo = st.sidebar.number_input("Volume Medio Minimo (30g)", min_value=100000, value=500000, step=100000)
cambio_eur_usd = st.sidebar.number_input("Tasso Cambio EUR/USD", min_value=0.5000, max_value=2.0000, value=cambio_default, step=0.0050, format="%.4f")

# --- CARICAMENTO TICKER ---
@st.cache_data
def carica_tickers(percorso_csv="nyse-listed.csv"):
    df_csv = pd.read_csv(percorso_csv)
    simboli = df_csv["ACT Symbol"].dropna().astype(str).str.strip()
    simboli_puliti = simboli[simboli.str.match(r'^[A-Z]{1,5}$')].unique().tolist()
    nomi_dict = dict(zip(df_csv["ACT Symbol"].str.strip(), df_csv["Company Name"].str.strip()))
    return simboli_puliti, nomi_dict

def calcola_metriche_base(df):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['Vol_Avg30'] = df['Volume'].rolling(window=30).mean()
    return df.iloc[-1]

@st.cache_data(ttl=3600)
def scarica_fondamentali_singolo(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        mcap = info.get("marketCap")
        mcap_b = round(mcap / 1e9, 2) if mcap else None
        
        pe_trailing = info.get("trailingPE")
        pe_forward = info.get("forwardPE")
        rev_growth = info.get("revenueGrowth")
        op_margin = info.get("operatingMargins")
        roe = info.get("returnOnEquity")
        debt_eq = info.get("debtToEquity")
        settore = info.get("sector", "N/D")
        
        # Data prossimi utili (timestamp unix in secondi)
        earnings_ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        earnings_str = datetime.fromtimestamp(earnings_ts).strftime("%d/%m/%Y") if earnings_ts else "N/D"

        return {
            "Ticker": ticker,
            "Settore": settore,
            "Cap. (Mld $)": mcap_b,
            "P/E Attuale": round(pe_trailing, 1) if pe_trailing else None,
            "P/E Stimato": round(pe_forward, 1) if pe_forward else None,
            "Crescita Ricavi (%)": round(rev_growth * 100, 1) if rev_growth is not None else None,
            "Margine Op. (%)": round(op_margin * 100, 1) if op_margin is not None else None,
            "ROE (%)": round(roe * 100, 1) if roe is not None else None,
            "Debito/Patr. (%)": round(debt_eq, 1) if debt_eq is not None else None,
            "Prossimi Utili": earnings_str
        }
    except Exception:
        return {"Ticker": ticker, "Settore": "Errore recupero dati"}

try:
    tickers, nomi_aziende = carica_tickers()
except FileNotFoundError:
    st.error("⚠️ File `nyse-listed.csv` non trovato.")
    st.stop()

if "dati_mercato" not in st.session_state:
    st.session_state.dati_mercato = []

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    avvia_download = st.button("🚀 1. Scarica / Aggiorna Dati Tecnici NYSE", type="primary", use_container_width=True)
with col_btn2:
    if st.session_state.dati_mercato:
        st.success(f"In memoria: {len(st.session_state.dati_mercato)} titoli")

if avvia_download:
    dati_raccolti = []
    barra = st.progress(0, text="Download dati tecnici in corso...")
    batch_size = 80
    totale = len(tickers)

    for i in range(0, totale, batch_size):
        batch = tickers[i:i + batch_size]
        progresso = min((i + batch_size) / totale, 1.0)
        barra.progress(progresso, text=f"Scaricati {len(dati_raccolti)} titoli validi su {min(i + batch_size, totale)}/{totale}...")

        try:
            dati_batch = yf.download(batch, period="2mo", interval="1d", group_by="ticker", progress=False, threads=True)
            time.sleep(0.8)
        except Exception:
            time.sleep(1.5)
            continue

        if dati_batch.empty:
            continue

        for ticker in batch:
            try:
                if isinstance(dati_batch.columns, pd.MultiIndex):
                    if ticker not in dati_batch.columns.get_level_values(0):
                        continue
                    df = dati_batch[ticker].dropna(how="all").copy()
                else:
                    df = dati_batch.dropna(how="all").copy()

                if df.empty or len(df) < 32:
                    continue

                oggi = calcola_metriche_base(df)
                prezzo  = float(oggi['Close'])
                ema9    = float(oggi['EMA9'])
                rsi     = float(oggi['RSI'])
                atr     = float(oggi['ATR'])
                vol     = float(oggi['Volume'])
                vol_avg = float(oggi['Vol_Avg30'])

                if any(pd.isna(x) for x in [prezzo, ema9, rsi, atr, vol, vol_avg]) or vol_avg == 0:
                    continue

                dati_raccolti.append({
                    "Ticker": ticker,
                    "Azienda": nomi_aziende.get(ticker, "")[:35],
                    "Prezzo": prezzo,
                    "EMA9": ema9,
                    "RSI": rsi,
                    "ATR": atr,
                    "Vol": vol,
                    "Vol_Avg30": vol_avg,
                    "RelVol": vol / vol_avg
                })
            except Exception:
                continue

    barra.empty()
    st.session_state.dati_mercato = dati_raccolti
    if len(dati_raccolti) == 0:
        st.error("🚫 Yahoo Finance non ha restituito dati. Attendi 2 minuti e riprova.")

# --- FILTRAGGIO TECNICO E ANALISI FONDAMENTALE ---
if st.session_state.dati_mercato:
    df_all = pd.DataFrame(st.session_state.dati_mercato)
    
    step1 = df_all[(df_all["Prezzo"] >= prezzo_minimo_usd) & (df_all["Vol_Avg30"] >= volume_minimo)]
    step2 = step1[step1["Prezzo"] > step1["EMA9"]]
    step3 = step2[(step2["RSI"] >= rsi_range[0]) & (step2["RSI"] <= rsi_range[1])]
    step4 = step3[step3["RelVol"] >= rel_vol_min]

    with st.expander("📊 Diagnostica Filtri Tecnici", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("1. Scaricati OK", len(df_all))
        c2.metric("2. Prezzo & Liq.", len(step1))
        c3.metric("3. Prezzo > EMA9", len(step2))
        c4.metric("4. Range RSI", len(step3))
        c5.metric("5. RelVol Min", len(step4))

    candidati = []
    capitale_usd = capitale_eur * cambio_eur_usd
    rischio_usd = capitale_usd * rischio_max_pct
    allocazione_max_usd = capitale_usd * allocazione_max_pct

    for _, row in step4.iterrows():
        prezzo = row["Prezzo"]
        atr = row["ATR"]
        stop_loss = prezzo - (moltiplicatore_atr * atr)
        rischio_per_azione = prezzo - stop_loss
        if rischio_per_azione <= 0:
            continue

        azioni = min(int(rischio_usd // rischio_per_azione), int(allocazione_max_usd // prezzo))
        if azioni < 1:
            continue

        controvalore_usd = azioni * prezzo
        rischio_effettivo_usd = azioni * rischio_per_azione
        take_profit = prezzo + (rischio_per_azione * rapporto_rr)
        profitto_potenziale_usd = azioni * (take_profit - prezzo)
        var_tp_pct = ((take_profit - prezzo) / prezzo) * 100

        candidati.append({
            "Ticker": row["Ticker"],
            "Azienda": row["Azienda"],
            "Prezzo (€)": round(prezzo / cambio_eur_usd, 2),
            "RSI": round(row["RSI"], 1),
            "RelVol": round(row["RelVol"], 2),
            "Azioni": azioni,
            "Investito (€)": round(controvalore_usd / cambio_eur_usd, 2),
            "Stop Loss (€)": round(stop_loss / cambio_eur_usd, 2),
            "Take Profit (€)": round(take_profit / cambio_eur_usd, 2),
            "Target (%)": round(var_tp_pct, 1),
            "Rischio (€)": round(rischio_effettivo_usd / cambio_eur_usd, 2),
            "Profitto Est. (€)": round(profitto_potenziale_usd / cambio_eur_usd, 2)
        })

    if not candidati:
        st.warning("Nessun titolo supera i filtri tecnici con queste impostazioni. Prova ad abbassare il Volume Relativo nella barra laterale.")
    else:
        df_risultati = pd.DataFrame(candidati).sort_values(by="RelVol", ascending=False).reset_index(drop=True)
        st.subheader(f"🎯 {len(df_risultati)} Candidati Tecnici (ordinati per Volume Relativo)")
        st.dataframe(df_risultati, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🏢 Analisi Fondamentale dei Candidati")
        
        if st.button("🔍 2. Analizza Fondamentali di questi titoli", use_container_width=True):
            lista_ticker_filtrati = df_risultati["Ticker"].tolist()
            dati_fond = []
            prog_fond = st.progress(0, text="Recupero dati di bilancio e trimestrali...")
            
            for idx, tk in enumerate(lista_ticker_filtrati, 1):
                prog_fond.progress(idx / len(lista_ticker_filtrati), text=f"Analisi fondamentale: {tk} ({idx}/{len(lista_ticker_filtrati)})...")
                dati_fond.append(scarica_fondamentali_singolo(tk))
                time.sleep(0.3)
            
            prog_fond.empty()
            df_fond = pd.DataFrame(dati_fond)
            
            # Unione tabella tecnica + fondamentale
            df_completo = pd.merge(df_risultati, df_fond, on="Ticker", how="left")
            st.dataframe(df_completo, use_container_width=True, hide_index=True)

            csv_completo = df_completo.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica Tabella Completa (Tecnica + Fondamentale) in CSV",
                data=csv_completo,
                file_name="screener_nyse_completo.csv",
                mime="text/csv"
            )
