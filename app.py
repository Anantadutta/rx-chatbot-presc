"""
Prescription Reader chatbot.

    streamlit run app.py

Upload a handwritten prescription -> get the patient's full record as JSON.
"""
import json
import os
import time
from pathlib import Path

# Use Windows' trusted certificates, so antivirus / office / college networks
# that inspect HTTPS don't cause "CERTIFICATE_VERIFY_FAILED" errors.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

# Load API keys from api_keys.txt (so you never have to paste them).
_keys_file = Path(__file__).parent / "api_keys.txt"
if _keys_file.exists():
    for _line in _keys_file.read_text(encoding="utf-8-sig").splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            if _v.strip() and "paste" not in _v.lower():
                os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

import hmac

import streamlit as st

# On Streamlit Cloud, keys come from the app's "Secrets" settings.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _v.strip():
            os.environ.setdefault(_k, _v.strip())
except Exception:
    pass  # no secrets file locally - that's fine

from reader import read_prescription
from records import list_patients, process_prescription, storage_name

st.set_page_config(page_title="Prescription Reader", page_icon="💊", layout="centered")


# ------------------------------------------------------------- password
def _password_ok() -> bool:
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return True  # no password set (e.g. running on your own computer)
    if st.session_state.get("authenticated"):
        return True
    st.title("💊 Prescription Reader")
    pw = st.text_input("Password", type="password")
    if st.button("Log in", type="primary"):
        if hmac.compare_digest(pw.encode(), expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")
    return False


if not _password_ok():
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider_label = st.radio(
        "AI that reads the handwriting",
        ["Gemini (free tier)", "Claude (paid, most accurate)"],
        index=1 if os.getenv("ANTHROPIC_API_KEY") and not os.getenv("GEMINI_API_KEY") else 0,
    )
    provider = "gemini" if provider_label.startswith("Gemini") else "claude"
    env_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    if os.getenv(env_name):
        api_key = os.getenv(env_name)  # never sent to the browser
        st.success("API key loaded from settings ✓")
    else:
        api_key = st.text_input("Gemini API key" if provider == "gemini" else "Anthropic API key",
                                type="password")
    if provider == "gemini":
        st.caption("On Gemini's free tier Google may use uploads to improve its products, "
                   "so test with sample prescriptions, not real patients.")
    st.divider()
    try:
        saved = list_patients()
    except Exception as e:
        saved = []
        st.error(f"Can't reach the records storage: {str(e)[:150]}")
    st.subheader(f"Saved patients ({len(saved)})")
    st.caption(f"Stored in {storage_name()}")
    if not saved:
        st.caption("None yet. Every prescription you upload is saved automatically.")
    for p in saved:
        st.caption(f"{p['patient_code']} · {p['name']} · {p['visits']} visit(s)")

st.title("💊 Prescription Reader")
st.caption("Upload a doctor's handwritten prescription. I'll read it, save it to the "
           "patient's record, and show every visit so far as JSON.")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi! Upload a prescription photo or PDF below and I'll get to work.",
    }]

# --------------------------------------------------------------- history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "json" in msg:
            st.json(msg["json"], expanded=2)
            st.download_button("⬇️ Download JSON", json.dumps(msg["json"], indent=2),
                               file_name=msg["filename"], mime="application/json",
                               key=f"dl_{i}")


def summary_text(r: dict) -> str:
    reading = r["prescription_reading"]
    icon = {"new_patient": "🆕", "returning_patient": "🔁",
            "already_uploaded": "♻️", "not_saved": "⚠️"}.get(r["status"], "")
    lines = [f"{icon} **{r['message']}**"]
    s = r.get("visit_summary")
    if s:
        lines.append("")
        lines.append(f"**Visits: {s['total_visits']}**")
        for v in r["visits"]:
            doc = (v.get("doctor") or {}).get("name") or "Unknown doctor"
            here = "  ← this upload" if v["visit_number"] == s["this_upload_is_visit_number"] else ""
            lines.append(f"{v['visit_number']}. **{v.get('date') or 'date not readable'}** — "
                         f"{doc} — {v.get('diagnosis') or 'no diagnosis written'}{here}")
    meds = reading["medicines"]
    if meds:
        lines.append("\n**Medicines on this prescription (best match):**")
        for m in meds:
            best = m["best_match"] or m["as_written"]
            extra = f" — written as *{m['as_written']}*" if m["best_match"] and \
                m["best_match"].lower() != (m["as_written"] or "").lower() else ""
            lines.append(f"- **{best}** {m['dosage'] or ''} {m['frequency'] or ''} "
                         f"({m['match_score']:.0f}% match){extra}")
    if reading["illegible_parts"]:
        lines.append("\n_Couldn't read clearly: " + "; ".join(reading["illegible_parts"]) + "_")
    return "\n".join(lines)


# ----------------------------------------------------------------- input
uploaded = st.file_uploader("Prescription", type=["jpg", "jpeg", "png", "webp", "gif", "pdf"],
                            label_visibility="collapsed")

if uploaded and st.button("Read prescription", type="primary"):
    if not api_key:
        st.error("Add your API key in the sidebar first.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": f"📄 Uploaded **{uploaded.name}**"})
    with st.spinner("Reading the handwriting and saving to the records..."):
        try:
            rx = None
            for attempt in range(3):  # retry on slow / dropped connections
                try:
                    rx = read_prescription(uploaded.getvalue(), uploaded.name, api_key, provider)
                    break
                except Exception as err:
                    msg = str(err).lower()
                    if attempt == 2 or not any(w in msg for w in ("timed out", "timeout", "connection", "interrupted", "503", "overloaded")):
                        raise
                    time.sleep(3)
            result = process_prescription(rx, uploaded.getvalue(), uploaded.name)
            st.session_state.messages.append({
                "role": "assistant", "content": summary_text(result), "json": result,
                "filename": f"{os.path.splitext(uploaded.name)[0]}_record.json",
            })
        except Exception as e:  # show the error in the chat instead of crashing
            err = str(e)
            if "401" in err or "API_KEY_INVALID" in err or "invalid x-api-key" in err:
                hint = ("That API key was rejected. Check that you picked the right AI in the "
                        "sidebar (a Gemini key starts with **AIza**, a Claude key with **sk-ant-**).")
            elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                hint = "Free-tier limit reached. Wait a minute (or until tomorrow) and try again."
            elif "CERTIFICATE_VERIFY_FAILED" in err:
                hint = ("Your network or antivirus is blocking the secure connection. "
                        "Turn off HTTPS/web scanning in your antivirus, or try a mobile hotspot.")
            elif "timed out" in err.lower() or "connection" in err.lower() or "DEADLINE" in err:
                hint = "The internet connection is too slow or dropped. Check your Wi-Fi and try again."
            else:
                hint = "Something went wrong."
            st.session_state.messages.append({"role": "assistant",
                                              "content": f"{hint}\n\n`{err[:300]}`"})
    st.rerun()
