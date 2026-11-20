"""
Patient records stored as JSON - one file per patient in the `patients` folder.

Every prescription uploaded in the chat is read by the AI and then:
  * if the patient is new          -> a new patient file is created (visit 1)
  * if the patient is already known -> the visit is added to their file
  * if the same file was uploaded before -> nothing is added (no double counting)

A patient is "the same" when the name is the same or very similar (handwriting
spelling varies: Chaurasiya / Chourasiya) AND the age does not clearly differ.
"""
import hashlib
import json
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path

from rapidfuzz import fuzz, process

PATIENTS_DIR = Path(__file__).parent / "patients"
NAME_MATCH_THRESHOLD = 85   # 0-100: how similar two names must be to be the same patient
AGE_TOLERANCE = 2           # years of difference still treated as the same person
MEDICINE_MATCH_THRESHOLD = 70

_lock = threading.Lock()

COMMON_MEDICINES = [
    "Paracetamol", "Ibuprofen", "Diclofenac", "Aceclofenac", "Aspirin", "Cetirizine",
    "Levocetirizine", "Montelukast", "Amoxicillin", "Amoxicillin + Clavulanic acid",
    "Azithromycin", "Ciprofloxacin", "Doxycycline", "Metronidazole", "Pantoprazole",
    "Omeprazole", "Rabeprazole", "Ranitidine", "Domperidone", "Ondansetron",
    "Metformin", "Glimepiride", "Gliclazide", "Sitagliptin", "Insulin glargine",
    "Amlodipine", "Telmisartan", "Losartan", "Atenolol", "Metoprolol", "Atorvastatin",
    "Rosuvastatin", "Clopidogrel", "Levothyroxine", "Prednisolone", "Salbutamol",
    "Vitamin D3", "Calcium carbonate", "Iron + Folic acid", "Multivitamin",
    "Hydrocortisone cream", "Dolo 650", "Crocin", "Augmentin", "Pan 40", "Pan-D",
    "Cefixime", "Chlorpheniramine", "Ambroxol", "Dextromethorphan", "ORS",
]


# ------------------------------------------------------------------ helpers
def _clean_name(name) -> str:
    s = str(name or "").lower()
    s = re.sub(r"\b(mr|mrs|ms|miss|master|baby|smt|shri|sri|dr)\b\.?", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    return " ".join(s.split())


def parse_age(text) -> int | None:
    """'28', '28 Y', '28yrs', '28/M', '6 months' -> whole years (None if unreadable)."""
    if text is None:
        return None
    s = str(text).lower()
    n = re.search(r"\d+(\.\d+)?", s)
    if not n:
        return None
    value = float(n.group())
    if re.search(r"\d\s*(m\b|mo|mon|month)", s):
        return int(value // 12)
    if re.search(r"\d\s*(d\b|day|w\b|wk|week)", s):
        return 0
    return int(value) if 0 <= value < 130 else None


def _to_date(s) -> date | None:
    try:
        return date.fromisoformat(str(s)[:10])
    except (TypeError, ValueError):
        return None


def _years_between(a: date | None, b: date | None) -> int:
    if not a or not b:
        return 0
    return b.year - a.year - ((b.month, b.day) < (a.month, a.day))


def _age_on(patient: dict, on: date | None) -> int | None:
    """Patient's estimated age on a given date, from the last age written on a prescription."""
    if patient.get("age_years") is None:
        return None
    recorded = _to_date(patient.get("age_recorded_on"))
    if not recorded or not on:
        return patient["age_years"]
    return patient["age_years"] + _years_between(recorded, on) if on >= recorded \
        else patient["age_years"] - _years_between(on, recorded)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40] or "Unknown"


# ------------------------------------------------------------------ storage
# If MONGODB_URI is set (online deployment), each patient is one JSON document in
# MongoDB (database "rx_chatbot", collection "patients"). Otherwise each patient is
# one JSON file in the local "patients" folder.
_mongo_collection = None


def _collection():
    global _mongo_collection
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    if _mongo_collection is None:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=15000)
        db = client[os.getenv("MONGODB_DB", "rx_chatbot")]
        _mongo_collection = db[os.getenv("MONGODB_COLLECTION", "patients")]
        _mongo_collection.create_index("patient_code", unique=True)
    return _mongo_collection


def storage_name() -> str:
    return "MongoDB (online)" if os.getenv("MONGODB_URI") else f"folder: {PATIENTS_DIR.name}/"


def _load_all() -> list[dict]:
    col = _collection()
    if col is not None:
        return list(col.find({}, {"_id": 0}).sort("patient_code", 1))
    PATIENTS_DIR.mkdir(exist_ok=True)
    out = []
    for f in sorted(PATIENTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = f
            out.append(data)
        except (json.JSONDecodeError, OSError):
            continue  # skip a damaged file instead of crashing
    return out


def _save(patient: dict) -> str:
    col = _collection()
    if col is not None:
        doc = {k: v for k, v in patient.items() if k not in ("_file", "_id")}
        col.replace_one({"patient_code": doc["patient_code"]}, doc, upsert=True)
        return f"MongoDB: {doc['patient_code']}"
    PATIENTS_DIR.mkdir(exist_ok=True)
    path = patient.pop("_file", None) or PATIENTS_DIR / f"{patient['patient_code']}_{_slug(patient['name'])}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(patient, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a crash never leaves half a file
    patient["_file"] = path
    return f"patients/{path.name}"


def _next_code(patients: list[dict]) -> str:
    nums = [int(p["patient_code"][2:]) for p in patients
            if str(p.get("patient_code", "")).startswith("P-") and p["patient_code"][2:].isdigit()]
    return f"P-{max(nums, default=1000) + 1}"


def list_patients() -> list[dict]:
    """Short list of all saved patients (for the sidebar)."""
    return [{"patient_code": p["patient_code"], "name": p["name"], "visits": len(p.get("visits", []))}
            for p in _load_all()]


# ----------------------------------------------------------------- matching
def find_patient(patients: list[dict], rx_patient: dict, rx_date) -> tuple[dict | None, dict]:
    """Best matching saved patient (or None) and an explanation of why."""
    name = _clean_name(rx_patient.get("name"))
    if not name:
        return None, {"reason": "no patient name could be read"}
    rx_age = parse_age(rx_patient.get("age"))
    on = _to_date(rx_date) or date.today()

    best, best_info = None, {"reason": "no saved patient with a similar name", "score": 0}
    for p in patients:
        names = [p["name"]] + p.get("other_spellings", [])
        score = max(fuzz.token_sort_ratio(name, _clean_name(n)) for n in names)
        if score < NAME_MATCH_THRESHOLD:
            continue
        known_age = _age_on(p, on)
        if rx_age is not None and known_age is not None and abs(rx_age - known_age) > AGE_TOLERANCE:
            info = {"reason": f"similar name ({score:.0f}%) but age differs "
                              f"({rx_age} vs ~{known_age}) - treated as a different person",
                    "score": score}
            if best is None:
                best_info = info
            continue
        rx_gender = str(rx_patient.get("gender") or "").strip().lower()[:1]
        p_gender = str(p.get("gender") or "").strip().lower()[:1]
        if rx_gender and p_gender and rx_gender != p_gender:
            continue  # different gender -> different person
        if best is None or score > best_info.get("score", 0):
            age_note = "age consistent" if rx_age is not None and known_age is not None else "age not compared"
            best, best_info = p, {"reason": f"name similarity {score:.0f}%, {age_note}", "score": score}
    return best, best_info


def match_medicines(rx_medicines: list[dict], known_extra: set[str]) -> list[dict]:
    known = sorted(set(COMMON_MEDICINES) | known_extra)
    out = []
    for m in rx_medicines or []:
        readings = [m.get("name") or ""] + list(m.get("alternatives") or [])
        best_name, best_score = None, 0.0
        for reading in filter(None, readings):
            hit = process.extractOne(reading, known, scorer=fuzz.WRatio)
            if hit and hit[1] > best_score:
                best_name, best_score = hit[0], hit[1]
        out.append({
            "as_written": m.get("name"),
            "best_match": best_name if best_score >= MEDICINE_MATCH_THRESHOLD else None,
            "match_score": round(best_score, 1),
            "reading_confidence": m.get("confidence"),
            "alternatives": m.get("alternatives") or [],
            "dosage": m.get("dosage"),
            "frequency": m.get("frequency"),
            "duration": m.get("duration"),
        })
    return out


# ------------------------------------------------------------------- output
def _history(p: dict, this_visit_id: str | None) -> dict:
    visits = sorted(p.get("visits", []),
                    key=lambda v: (v.get("date") or "9999", v.get("uploaded_at") or ""))
    numbered, doctors = [], {}
    for i, v in enumerate(visits, 1):
        numbered.append({"visit_number": i, **{k: v[k] for k in v if k != "file_hash"}})
        dname = (v.get("doctor") or {}).get("name") or "Unknown doctor"
        # "Dr. S. Khan" and "Dr S Khan" are the same doctor
        key = next((k for k in doctors
                    if fuzz.token_sort_ratio(_clean_name(k), _clean_name(dname)) >= NAME_MATCH_THRESHOLD),
                   dname)
        d = doctors.setdefault(key, {"doctor": key, "times_seen": 0, "dates": []})
        d["times_seen"] += 1
        d["dates"].append(v.get("date") or "date not readable")
    this_no = next((n["visit_number"] for n in numbered if n["visit_id"] == this_visit_id), None)
    age_now = _age_on(p, date.today())
    return {
        "patient": {
            "patient_code": p["patient_code"], "name": p["name"],
            "other_spellings": p.get("other_spellings", []),
            "age": age_now,
            "age_source": (f"estimated from age written on prescription of {p.get('age_recorded_on')}"
                           if age_now is not None else None),
            "gender": p.get("gender"), "phone": p.get("phone"),
            "first_seen": p.get("created_at"),
        },
        "visit_summary": {
            "total_visits": len(numbered),
            "this_upload_is_visit_number": this_no,
            "first_visit": numbered[0]["date"] if numbered else None,
            "last_visit": numbered[-1]["date"] if numbered else None,
            "visit_dates": [v["date"] for v in numbered],
            "doctors_seen": list(doctors.values()),
        },
        "visits": numbered,
    }


# --------------------------------------------------------------------- main
def process_prescription(rx: dict, file_bytes: bytes, filename: str) -> dict:
    """Save a freshly read prescription and return the JSON shown to the user."""
    with _lock:
        patients = _load_all()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        p_rx = rx.get("patient") or {}
        known_meds = {m.get("name") for p in patients for v in p.get("visits", [])
                      for m in v.get("medicines", []) if m.get("name")}
        medicines = match_medicines(rx.get("medicines"), known_meds)

        reading = {
            "patient": p_rx, "doctor": rx.get("doctor"), "date": rx.get("date"),
            "diagnosis": rx.get("diagnosis"), "medicines": medicines, "advice": rx.get("advice"),
            "field_confidence": rx.get("field_confidence"),
            "illegible_parts": rx.get("illegible_parts") or [],
        }

        # 1) exact same file uploaded before -> don't count again
        for p in patients:
            for v in p.get("visits", []):
                if v.get("file_hash") == file_hash:
                    return {"status": "already_uploaded",
                            "message": f"This prescription was already uploaded on "
                                       f"{v.get('uploaded_at', '')[:10]}. It was not counted again.",
                            "prescription_reading": reading, **_history(p, v["visit_id"])}

        # 2) no name -> cannot be filed
        if not _clean_name(p_rx.get("name")):
            return {"status": "not_saved",
                    "message": "The patient's name could not be read, so this prescription "
                               "was not saved. Try a clearer photo.",
                    "prescription_reading": reading}

        patient, why = find_patient(patients, p_rx, rx.get("date"))
        is_new = patient is None
        now = datetime.now().isoformat(timespec="seconds")
        if is_new:
            patient = {"patient_code": _next_code(patients), "name": p_rx["name"].strip().title(),
                       "other_spellings": [], "gender": p_rx.get("gender"), "phone": p_rx.get("phone"),
                       "age_years": None, "age_recorded_on": None, "created_at": now, "visits": []}
        else:
            written = p_rx["name"].strip().title()
            if _clean_name(written) != _clean_name(patient["name"]) and \
                    written not in patient.setdefault("other_spellings", []):
                patient["other_spellings"].append(written)
            for field in ("gender", "phone"):
                if p_rx.get(field) and not patient.get(field):
                    patient[field] = p_rx[field]

        # keep the most recent written age (to estimate age later)
        age = parse_age(p_rx.get("age"))
        if age is not None and (patient.get("age_recorded_on") is None or
                                (rx.get("date") or "") >= (patient.get("age_recorded_on") or "")):
            patient["age_years"], patient["age_recorded_on"] = age, rx.get("date") or now[:10]

        visit_id = f"{patient['patient_code']}-{len(patient['visits']) + 1}-{file_hash[:8]}"
        patient["visits"].append({
            "visit_id": visit_id,
            "date": rx.get("date"),
            "uploaded_at": now,
            "source_file": filename,
            "file_hash": file_hash,
            "doctor": rx.get("doctor") or {},
            "diagnosis": rx.get("diagnosis"),
            "advice": rx.get("advice"),
            "age_at_visit": p_rx.get("age"),
            "medicines": [{"name": m["best_match"] or m["as_written"], "as_written": m["as_written"],
                           "dosage": m["dosage"], "frequency": m["frequency"],
                           "duration": m["duration"]} for m in medicines],
        })
        saved_to = _save(patient)

        hist = _history(patient, visit_id)
        n = hist["visit_summary"]["total_visits"]
        if is_new:
            status, msg = "new_patient", (f"New patient {patient['name']} ({patient['patient_code']}). "
                                          f"This is their 1st recorded visit.")
        else:
            status, msg = "returning_patient", (f"{patient['name']} ({patient['patient_code']}) has now "
                                                f"visited {n} times ({why['reason']}).")
        return {"status": status, "message": msg, "saved_to": saved_to,
                "prescription_reading": reading, **hist}
