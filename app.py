import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(
    page_title="AlphaPulse Screener NYSE",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaPulse — Screener NYSE (EUR)")
st.markdown("Strategia **Long Trend & Momentum** (`Prezzo > EMA9`, `RSI`, `Volume`) con gestione automatica del rischio e conversione in Euro.")

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
st.sidebar.subheader("🔍 Filtri Strategia & Liquidità")
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
    st.info(
        f"📋 Listino: **{len(tickers)}** titoli NYSE | "
        f"Max per singolo trade: **€{capitale_eur * allocazione_max_pct:,.2f}** ({allocazione_max_pct*100:.0f}%) | "
        f"Cambio: **1 € = {cambio_eur_usd:.4f} $**"
    )
except FileNotFoundError:
    st.error("⚠️ File `nyse-listed.csv` non trovato. Caricalo nella stessa cartella di `app.py` su GitHub.")
    st.stop()

# --- AVVIO SCANSIONE ---
if st.button("🚀 Avvia Scansione Completa NYSE", type="primary", use_container_width=True):
    candidati = []
    scaricati_ok = 0
    capitale_usd = capitale_eur * cambio_eur_usd
    rischio_usd = capitale_usd * rischio_max_pct
    allocazione_max_usd = capitale_usd * allocazione_max_pct

    barra = st.progress(0, text="Inizializzazione scansione...")
    
    batch_size = 50
    totale = len(tickers)

    for i in range(0, totale, batch_size):
        batch = tickers[i:i + batch_size]
        progresso = min((i + batch_size) / totale, 1.0)
        barra.progress(
            progresso,
            text=f"Analisi: {min(i + batch_size, totale)}/{totale} | Scaricati validi: {scaricati_ok} | Trovati: {len(candidati)}"
        )

        try:
            dati_batch = yf.download(batch, period="3mo", interval="1d", group_by="ticker", progress=False, threads=False)
            time.sleep(1.2)  # Pausa anti-blocco per evitare rate-limit di Yahoo Finance
        except Exception:
            time.sleep(2.0)
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

                if df.empty or len(df) < 35:
                    continue

                scaricati_ok += 1
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

                rel_vol = vol / vol_avg

                if (prezzo >= prezzo_minimo_usd and
                    vol_avg >= volume_minimo and
                    prezzo > ema9 and
                    rsi_range[0] <= rsi <= rsi_range[1] and
                    rel_vol >= rel_vol_min):

                    stop_loss = prezzo - (moltiplicatore_atr * atr)
                    rischio_per_azione = prezzo - stop_loss
                    if rischio_per_azione <= 0:
                        continue

                    azioni_da_rischio = int(rischio_usd // rischio_per_azione)
                    azioni_da_allocazione = int(allocazione_max_usd // prezzo)
                    
                    azioni = min(azioni_da_rischio, azioni_da_allocazione)
                    if azioni < 1:
                        continue

                    controvalore_usd = azioni * prezzo
                    rischio_effettivo_usd = azioni * rischio_per_azione
                    
                    take_profit = prezzo + (rischio_per_azione * rapporto_rr)
                    profitto_potenziale_usd = azioni * (take_profit - prezzo)
                    var_tp_pct = ((take_profit - prezzo) / prezzo) * 100

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
                        "Rischio (€)": round(rischio_effettivo_usd / cambio_eur_usd, 2),
                        "Profitto Est. (€)": round(profitto_potenziale_usd / cambio_eur_usd, 2)
                    })
            except Exception:
                continue

    barra.empty()

    if scaricati_ok == 0:
        st.error("🚫 Yahoo Finance ha temporaneamente bloccato le richieste dal server cloud (0 titoli scaricati). Attendi 2-3 minuti e riprova.")
    elif not candidati:
        st.warning(f"Analizzati **{scaricati_ok}** titoli validi, ma nessuno soddisfa tutti i criteri oggi. Prova ad abbassare il **Volume Relativo Minimo** a `0.8` nella barra laterale.")
    else:
        df_risultati = pd.DataFrame(candidati).sort_values(by="RelVol", ascending=False).reset_index(drop=True)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Titoli Idonei", f"{len(df_risultati)} / {scaricati_ok}")
        col2.metric("Tetto per Trade", f"€ {capitale_eur * allocazione_max_pct:,.2f}")
        col3.metric("Rischio Max Teorico", f"€ {capitale_eur * rischio_max_pct:,.2f}")
        col4.metric("R:R Impostato", f"1 : {rapporto_rr:.1f}")

        st.subheader("🎯 Migliori Opportunità (ordinate per Volume Relativo)")
        st.dataframe(df_risultati, use_container_width=True, hide_index=True)

        csv_export = df_risultati.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica risultati in CSV",
            data=csv_export,
            file_name="screener_nyse_eur.csv",
            mime="text/csv"
        )
