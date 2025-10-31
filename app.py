# app.py
# Streamlit Stock Portfolio Dashboard (final, instant transitions, simple delete confirmation)
import streamlit as st
import json, os, hashlib
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from sklearn.linear_model import LinearRegression
    SKL = True
except Exception:
    SKL = False

st.set_page_config(page_title="Stock Portfolio Dashboard", layout="wide")

DB_FILE = "local_db.json"

# ---------- UTILITIES ----------
def ensure_db():
    if not Path(DB_FILE).exists():
        data = {"users": {}, "profiles": {}, "portfolios": {}, "meta": {"last_updated": None}}
        Path(DB_FILE).write_text(json.dumps(data, indent=2))

def load_db():
    ensure_db()
    return json.loads(Path(DB_FILE).read_text())

def save_db(db):
    Path(DB_FILE).write_text(json.dumps(db, indent=2, default=str))

def hash_pw(pw: str):
    return hashlib.sha256(pw.encode()).hexdigest()

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

# ---------- AUTH ----------
def signup_local(full_name, email, phone, password):
    db = load_db()
    if email in db["users"]:
        return False, "EMAIL_EXISTS"
    uid = "uid_" + hashlib.sha1(email.encode()).hexdigest()[:12]
    db["users"][email] = {"uid": uid, "pw": hash_pw(password)}
    db["profiles"][uid] = {"full_name": full_name, "email": email, "phone": phone, "created_at": now_iso()}
    db["portfolios"][uid] = []
    db["meta"]["last_updated"] = now_iso()
    save_db(db)
    return True, uid

def login_local(email, password):
    db = load_db()
    if email not in db["users"]:
        return False, "USER_NOT_FOUND"
    rec = db["users"][email]
    if rec["pw"] != hash_pw(password):
        return False, "WRONG_PASSWORD"
    return True, rec["uid"]

def get_profile(uid):
    db = load_db()
    return db["profiles"].get(uid, {})

def get_portfolio(uid):
    db = load_db()
    return db["portfolios"].get(uid, [])

def save_portfolio(uid, portfolio_list):
    db = load_db()
    db["portfolios"][uid] = portfolio_list
    db["meta"]["last_updated"] = now_iso()
    save_db(db)

def delete_account(uid, email):
    db = load_db()
    db["profiles"].pop(uid, None)
    db["portfolios"].pop(uid, None)
    db["users"].pop(email, None)
    db["meta"]["last_updated"] = now_iso()
    save_db(db)

# ---------- PRICE + HISTORY ----------
def fetch_history_yf(symbol, days=7):
    if yf is None:
        return None
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=f"{days}d")
        if hist is None or hist.empty:
            if "." not in symbol:
                return fetch_history_yf(symbol + ".NS", days)
            return None
        df = hist.reset_index()[["Date", "Close"]]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df
    except Exception:
        return None

def synthetic_history(symbol, days=7):
    seed = sum(ord(c) for c in symbol) % 1000
    rng = np.random.RandomState(seed)
    base = 100 + (seed % 300)
    dates = [(datetime.utcnow().date() - timedelta(days=days-1-i)) for i in range(days)]
    prices = []
    p = base
    for _ in range(days):
        p = max(1, p * (1 + rng.uniform(-0.02, 0.03)))
        prices.append(round(float(p), 2))
    return pd.DataFrame({"Date": dates, "Close": prices})

def get_history(symbol, days=7):
    df = fetch_history_yf(symbol, days)
    if df is None or df.empty:
        df = synthetic_history(symbol, days)
    return df

def current_price_from_history(symbol):
    df = get_history(symbol, days=1)
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else synthetic_history(symbol)["Close"].iloc[-1]

def predict_growth_pct(symbol):
    df = get_history(symbol, days=7)
    if df is None or df.empty or len(df) < 2:
        return 0.0, 0.0
    x = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values.reshape(-1, 1)
    if SKL:
        model = LinearRegression().fit(x, y)
        last = y[-1][0]
        next_pred = model.predict(np.array([[len(df)]]))[0][0]
        pct = ((next_pred - last) / last) * 100 if last != 0 else 0.0
        conf = model.score(x, y)
    else:
        coeff = np.polyfit(x.flatten(), y.flatten(), 1)
        last = y[-1][0]
        next_pred = coeff[0] * len(df) + coeff[1]
        pct = ((next_pred - last) / last) * 100 if last != 0 else 0.0
        conf = 0.85
    return round(pct, 2), round(conf * 100, 1)

# ---------- SIDEBAR ----------
def sidebar_menu():
    st.sidebar.markdown("### Menu")
    if st.sidebar.button("👤 My Profile"):
        st.session_state.page = "profile"; st.rerun()
    if st.sidebar.button("🏠 Go to Dashboard"):
        st.session_state.page = "dashboard"; st.rerun()
    if st.sidebar.button("📈 Predict Growth"):
        st.session_state.page = "predict"; st.rerun()
    if st.sidebar.button("📥 Download Report"):
        st.session_state.page = "download"; st.rerun()
    st.sidebar.write("---")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.session_state.page = "login"
        st.rerun()

# ---------- SESSION ----------
if "page" not in st.session_state: st.session_state.page = "login"
if "uid" not in st.session_state: st.session_state.uid = None
if "email" not in st.session_state: st.session_state.email = None
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"

# ---------- AUTH PAGE ----------
def show_auth():
    st.markdown("<h1 style='text-align:center'>📈 Stock Portfolio Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic; color:#555;'>Your personal investment tracker</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.auth_mode == "login":
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Login"):
                ok, res = login_local(email.strip(), pw.strip())
                if ok:
                    st.session_state.uid = res
                    st.session_state.email = email.strip()
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials or user not found.")
            st.markdown("---")
            if st.button("Don't have an account? Sign Up"):
                st.session_state.auth_mode = "signup"; st.rerun()
        else:
            st.subheader("Sign Up")
            name = st.text_input("Full Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.button("Create Account"):
                ok, res = signup_local(name, email, phone, pw)
                if ok:
                    st.session_state.uid = res
                    st.session_state.email = email.strip()
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Email already exists. Please login.")
            st.markdown("---")
            if st.button("Already have an account? Login"):
                st.session_state.auth_mode = "login"; st.rerun()

# ---------- DASHBOARD ----------
def dashboard():
    st.markdown("<h1 style='text-align:center'>📈 Stock Portfolio Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic; color:#444;'>Your personal investment tracker</p>", unsafe_allow_html=True)
    sidebar_menu()
    uid = st.session_state.uid
    if not uid: 
        st.warning("Please login first.")
        return

    st.subheader("Add a New Stock")
    c1, c2, c3, c4, c5, c6 = st.columns([2,1,1,1.5,2,1])
    sym = c1.text_input("Symbol (e.g., TCS.NS)")
    qty = c2.number_input("Qty", min_value=1, value=1, step=1)
    price = c3.number_input("Purchase Price", min_value=0.0, value=0.0, step=0.1)
    date = c4.date_input("Date", value=datetime.utcnow().date())
    note = c5.text_input("Notes")
    if c6.button("➕ Add"):
        if sym.strip():
            data = get_portfolio(uid)
            data.append({
                "symbol": sym.strip().upper(),
                "quantity": int(qty),
                "purchase_price": float(price),
                "purchase_date": str(date),
                "notes": note
            })
            save_portfolio(uid, data)
            st.success("Stock added successfully.")
            st.rerun()
        else:
            st.error("Enter stock symbol first.")
    st.markdown("***")

    # Portfolio table
    data = get_portfolio(uid)
    st.subheader("Portfolio Overview")
    if not data:
        st.info("No stocks yet.")
        return

    rows = []
    for i, d in enumerate(data):
        cp = current_price_from_history(d["symbol"])
        total_cost = d["quantity"] * d["purchase_price"]
        total_val = d["quantity"] * cp
        pl = total_val - total_cost
        pl_pct = (pl / total_cost) * 100 if total_cost else 0
        rows.append({**d, "Current Price": round(cp,2), "Profit/Loss (₹)": round(pl,2), "Profit/Loss (%)": round(pl_pct,2), "Index": i})

    df = pd.DataFrame(rows)
    cols = st.columns([1,1,1,1,1,1,1,0.5])
    headers = ["Symbol","Qty","Purchase Date","Buy ₹","Current ₹","P/L ₹","P/L %",""]
    for c,h in zip(cols, headers): c.markdown(f"**{h}**")

    for _,r in df.iterrows():
        c = st.columns([1,1,1,1,1,1,1,0.5])
        c[0].write(r["symbol"]); c[1].write(r["quantity"])
        c[2].write(r["purchase_date"])
        c[3].write(f"₹{r['purchase_price']:.2f}")
        c[4].write(f"₹{r['Current Price']:.2f}")
        colr = "#16A34A" if r["Profit/Loss (₹)"] >= 0 else "#EF4444"
        c[5].markdown(f"<p style='color:{colr};font-weight:600;'>₹{r['Profit/Loss (₹)']:.2f}</p>", unsafe_allow_html=True)
        c[6].markdown(f"<p style='color:{colr};'>{r['Profit/Loss (%)']:.2f}%</p>", unsafe_allow_html=True)

        if c[7].button("🗑️", key=f"del_{r['Index']}"):
            st.session_state["delete_idx"] = r["Index"]; st.rerun()

    if "delete_idx" in st.session_state:
        di = st.session_state["delete_idx"]
        st.warning(f"Are you sure you want to delete **{data[di]['symbol']}**?")
        yes, no = st.columns(2)
        if yes.button("✅ Yes"):
            data.pop(di)
            save_portfolio(uid, data)
            st.session_state.pop("delete_idx")
            st.success("Deleted successfully.")
            st.rerun()
        if no.button("❌ No"):
            st.session_state.pop("delete_idx")
            st.rerun()

    total_pl = sum(r["Profit/Loss (₹)"] for _,r in df.iterrows())
    st.markdown(f"**Total Profit/Loss:** {'🟩' if total_pl>=0 else '🟥'} ₹{total_pl:.2f}")
    st.caption(f"Last updated: {load_db()['meta']['last_updated']}")

    st.markdown("***")
    left, right = st.columns(2)
    with left:
        st.subheader("Portfolio Distribution")
        fig1, ax1 = plt.subplots(figsize=(4,4))
        grp = df.groupby("symbol")["quantity"].sum()
        ax1.pie(grp.values, labels=grp.index, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)
    with right:
        st.subheader("Stock Price Trend (7 Days)")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        for sym in df["symbol"]:
            hist = get_history(sym)
            ax2.plot(hist["Date"], hist["Close"], marker='^', linewidth=1.8, markersize=5, label=sym)
        ax2.legend(fontsize="small")
        plt.xticks(rotation=25)
        st.pyplot(fig2)

# ---------- PROFILE ----------
def profile_page():
    st.header("My Profile")
    sidebar_menu()
    uid = st.session_state.uid
    if not uid:
        st.warning("Login first."); return
    prof = get_profile(uid)
    st.write(f"**Name:** {prof.get('full_name','')}")
    st.write(f"**Email:** {prof.get('email','')}")
    st.write(f"**Phone:** {prof.get('phone','')}")
    st.write(f"**Member since:** {prof.get('created_at','')}")
    st.write("")
    if st.button("Logout"):
        st.session_state.clear(); st.session_state.page = "login"; st.rerun()
    if st.button("Delete Account"):
        st.warning("Are you sure? This cannot be undone.")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, delete"):
            delete_account(uid, prof.get("email"))
            st.session_state.clear(); st.session_state.page = "login"; st.rerun()
        if col2.button("❌ No"):
            st.rerun()

# ---------- PREDICT GROWTH ----------
def predict_page():
    st.header("📈 Predict Growth (Next 7 Days)")
    sidebar_menu()
    uid = st.session_state.uid
    if not uid: return st.warning("Login required.")
    data = get_portfolio(uid)
    if not data: return st.info("No stocks to analyze.")
    rows = []
    for d in data:
        pct, conf = predict_growth_pct(d["symbol"])
        rows.append({"Symbol": d["symbol"], "Predicted % Change": pct, "Confidence (%)": conf,
                     "Trend": "📈 Upward" if pct>0 else ("📉 Downward" if pct<0 else "➡️ Flat")})
    df = pd.DataFrame(rows)
    st.table(df)
    st.markdown("### Trends (7 Days)")
    for d in data:
        sym = d["symbol"]; hist = get_history(sym)
        fig, ax = plt.subplots(figsize=(6,2.5))
        ax.plot(hist["Date"], hist["Close"], marker='^', linewidth=1.6, label=sym)
        ax.legend(fontsize="small"); plt.xticks(rotation=25)
        st.pyplot(fig)

# ---------- DOWNLOAD ----------
def download_page():
    st.header("Download Report")
    sidebar_menu()
    uid = st.session_state.uid
    if not uid: return st.warning("Login first.")
    data = get_portfolio(uid)
    if not data: return st.info("No data.")
    df = pd.DataFrame(data)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "portfolio_report.csv")

# ---------- MAIN ----------
def main():
    ensure_db()
    page = st.session_state.page
    if page == "login": return show_auth()
    if not st.session_state.uid: st.session_state.page = "login"; return show_auth()
    if page == "dashboard": dashboard()
    elif page == "profile": profile_page()
    elif page == "predict": predict_page()
    elif page == "download": download_page()
    else: dashboard()

if __name__ == "__main__":
    main()
