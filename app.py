import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="AlphaPulse Screener NYSE",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaPulse — Screener NYSE (EUR)")
st.markdown("Strategia **Long Trend & Momentum** (`Prezzo > EMA9`, `RSI 50-68`, `Volume > Media 30g`) con calcolo automatico di Position Size, Stop Loss (ATR) e Take Profit in Euro.")

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
target_cap_pct = st.sidebar.slider("Target Profitto sul Capitale (%)", min_value=1.0, max_value=10.0, value=4.0, step=0.5) / 100
moltiplicatore_atr = st.sidebar.slider("Moltiplicatore ATR (Stop Loss)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("🔍 Filtri Mercato & Cambio")
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

def calcola_indicatori(df):
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
    return df

try:
    tickers, nomi_aziende = carica_tickers()
    st.info(f"📋 Listino caricato: **{len(tickers)}** azioni ordinarie NYSE pronte per lo screening. | Cambio attivo: **1 € = {cambio_eur_usd:.4f} $**")
except FileNotFoundError:
    st.error("⚠️ File `nyse-listed.csv` non trovato. Caricalo nella stessa cartella di `app.py` su GitHub.")
    st.stop()

# --- AVVIO SCANSIONE ---
if st.button("🚀 Avvia Scansione Completa NYSE", type="primary", use_container_width=True):
    candidati = []
    capitale_usd = capitale_eur * cambio_eur_usd
    rischio_usd = capitale_usd * rischio_max_pct
    target_usd = capitale_usd * target_cap_pct

    barra = st.progress(0, text="Inizializzazione scansione...")
    stato_testo = st.empty()
    
    batch_size = 100
    totale = len(tickers)

    for i in range(0, totale, batch_size):
        batch = tickers[i:i + batch_size]
        progresso = min((i + batch_size) / totale, 1.0)
        barra.progress(progresso, text=f"Analisi in corso: {min(i + batch_size, totale)} / {totale} titoli... (Trovati: {len(candidati)})")

        try:
            dati_batch = yf.download(batch, period="3mo", interval="1d", group_by="ticker", progress=False, threads=True)
        except Exception:
            continue

        for ticker in batch:
            try:
                if ticker not in dati_batch.columns.get_level_values(0):
                    continue
                df = dati_batch[ticker].dropna(how="all").copy()
                if df.empty or len(df) < 35:
                    continue

                df = calcola_indicatori(df)
                oggi = df.iloc[-1]

                prezzo  = float(oggi['Close'])
                ema9    = float(oggi['EMA9'])
                rsi     = float(oggi['RSI'])
                atr     = float(oggi['ATR'])
                vol     = float(oggi['Volume'])
                vol_avg = float(oggi['Vol_Avg30'])

                if pd.isna(rsi) or pd.isna(atr) or pd.isna(vol_avg) or vol_avg == 0:
                    continue

                if (prezzo >= prezzo_minimo_usd and
                    vol_avg >= volume_minimo and
                    prezzo > ema9 and
                    50 <= rsi <= 68 and
                    vol > vol_avg):

                    stop_loss = prezzo - (moltiplicatore_atr * atr)
                    rischio_per_azione = prezzo - stop_loss
                    if rischio_per_azione <= 0:
                        continue

                    azioni = int(rischio_usd // rischio_per_azione)
                    controvalore_usd = azioni * prezzo

                    if controvalore_usd > capitale_usd:
                        azioni = int(capitale_usd // prezzo)
                        controvalore_usd = azioni * prezzo

                    if azioni < 1:
                        continue

                    rischio_effettivo_usd = azioni * rischio_per_azione
                    take_profit = prezzo + (target_usd / azioni)
                    var_tp_pct = ((take_profit - prezzo) / prezzo) * 100
                    rel_vol = vol / vol_avg

                    candidati.append({
                        "Ticker": ticker,
                        "Azienda": nomi_aziende.get(ticker, "")[:35],
                        "Prezzo (€)": round(prezzo / cambio_eur_usd, 2),
                        "RSI": round(rsi, 1),
                        "RelVol": round(rel_vol, 2),
                        "Azioni": azioni,
                        "Investito (€)": round(controvalore_usd / cambio_eur_usd, 2),
                        "Stop Loss (€)": round(stop_loss / cambio_eur_usd, 2),
                        "Take Profit (€)": round(take_profit / cambio_eur_usd, 2),
                        "Target (%)": round(var_tp_pct, 1),
                        "Rischio Max (€)": round(rischio_effettivo_usd / cambio_eur_usd, 2)
                    })
            except Exception:
                continue

    barra.empty()
    stato_testo.empty()

    if not candidati:
        st.warning("Nessun titolo NYSE soddisfa tutti i criteri con i parametri attuali.")
    else:
        df_risultati = pd.DataFrame(candidati).sort_values(by="RelVol", ascending=False).reset_index(drop=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Titoli Idonei Trovati", len(df_risultati))
        col2.metric("Rischio Max Impostato", f"€ {capitale_eur * rischio_max_pct:,.2f}")
        col3.metric("Target Profitto Impostato", f"€ {capitale_eur * target_cap_pct:,.2f}")

        st.subheader("🎯 Migliori Opportunità (ordinate per Volume Relativo)")
        st.dataframe(df_risultati, use_container_width=True, hide_index=True)

        csv_export = df_risultati.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica risultati in CSV",
            data=csv_export,
            file_name="screener_nyse_eur.csv",
            mime="text/csv"
        )
