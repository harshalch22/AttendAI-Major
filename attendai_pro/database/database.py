"""
database/database.py
--------------------
Complete SQLite database for AttendAI Pro.

Tables:
  - classes       : divisions/classes (e.g. CS-A, CS-B)
  - students      : students linked to a class
  - attendance    : one record per student per date per class
  - emotions      : emotion logs
  - mobile_usage  : mobile phone detection logs
  - head_down     : head down / distraction logs
  - alerts        : real-time alert log
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "attendai_pro.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    # ── Classes ──────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL UNIQUE,
            department TEXT NOT NULL,
            year       TEXT NOT NULL,
            created_at TEXT DEFAULT (date('now'))
        )
    """)

    # ── Students ─────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            roll_no    TEXT NOT NULL UNIQUE,
            class_id   INTEGER NOT NULL,
            created_at TEXT DEFAULT (date('now')),
            FOREIGN KEY(class_id) REFERENCES classes(id)
        )
    """)

    # ── Attendance ────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            name       TEXT NOT NULL,
            class_id   INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            date       TEXT NOT NULL,
            time       TEXT NOT NULL,
            status     TEXT DEFAULT 'Present',
            UNIQUE(student_id, date, class_id),
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
    """)

    # ── Emotions ──────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            class_name   TEXT NOT NULL,
            emotion      TEXT NOT NULL,
            confidence   REAL DEFAULT 0,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL
        )
    """)

    # ── Mobile usage ──────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS mobile_usage (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name  TEXT NOT NULL,
            class_name    TEXT NOT NULL,
            date          TEXT NOT NULL,
            first_seen    TEXT NOT NULL,
            last_seen     TEXT NOT NULL,
            total_seconds INTEGER DEFAULT 0
        )
    """)

    # ── Head down ─────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS head_down (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name  TEXT NOT NULL,
            class_name    TEXT NOT NULL,
            date          TEXT NOT NULL,
            time          TEXT NOT NULL,
            duration_sec  INTEGER DEFAULT 0
        )
    """)

    # ── Alerts ────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type   TEXT NOT NULL,
            student_name TEXT,
            class_name   TEXT,
            message      TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] All tables initialised.")


# ── Class helpers ─────────────────────────────────────────────────

def add_class(name, department, year):
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO classes (name, department, year) VALUES (?,?,?)",
            (name.strip(), department.strip(), year.strip())
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_classes():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM classes ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_class_by_id(class_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM classes WHERE id=?", (class_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_class(class_id):
    conn = get_conn()
    conn.execute("DELETE FROM classes WHERE id=?", (class_id,))
    conn.commit()
    conn.close()


# ── Student helpers ───────────────────────────────────────────────

def add_student(name, roll_no, class_id):
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO students (name, roll_no, class_id) VALUES (?,?,?)",
            (name.strip(), roll_no.strip(), class_id)
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_students_by_class(class_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT s.*, c.name as class_name FROM students s "
        "JOIN classes c ON s.class_id = c.id "
        "WHERE s.class_id=? ORDER BY s.name",
        (class_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_students():
    conn = get_conn()
    rows = conn.execute(
        "SELECT s.*, c.name as class_name FROM students s "
        "JOIN classes c ON s.class_id = c.id ORDER BY c.name, s.name"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_student_names_by_class(class_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT name FROM students WHERE class_id=?", (class_id,)
    ).fetchall()
    conn.close()
    return [r["name"] for r in rows]


def student_exists(name):
    conn = get_conn()
    row = conn.execute("SELECT id FROM students WHERE name=?", (name,)).fetchone()
    conn.close()
    return row is not None


def delete_student(student_id):
    conn = get_conn()
    conn.execute("DELETE FROM students WHERE id=?", (student_id,))
    conn.commit()
    conn.close()


# ── Attendance helpers ────────────────────────────────────────────

def mark_attendance(name, class_id, class_name, date, time):
    conn = get_conn()
    row = conn.execute("SELECT id FROM students WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        return False
    cursor = conn.execute(
        "INSERT OR IGNORE INTO attendance "
        "(student_id, name, class_id, class_name, date, time) VALUES (?,?,?,?,?,?)",
        (row["id"], name, class_id, class_name, date, time)
    )
    inserted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return inserted


def get_attendance(class_id=None, date_filter=None):
    conn = get_conn()
    q = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if class_id:
        q += " AND class_id=?"; params.append(class_id)
    if date_filter:
        q += " AND date=?"; params.append(date_filter)
    q += " ORDER BY time DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_dates():
    conn = get_conn()
    rows = conn.execute(
        "SELECT DISTINCT date FROM attendance ORDER BY date DESC"
    ).fetchall()
    conn.close()
    return [r["date"] for r in rows]


# ── Emotion helpers ───────────────────────────────────────────────

def log_emotion(student_name, class_name, emotion, confidence, date, time):
    conn = get_conn()
    conn.execute(
        "INSERT INTO emotions (student_name, class_name, emotion, confidence, date, time) "
        "VALUES (?,?,?,?,?,?)",
        (student_name, class_name, emotion, confidence, date, time)
    )
    conn.commit()
    conn.close()


def get_emotions(class_name=None, date_filter=None):
    conn = get_conn()
    q = "SELECT * FROM emotions WHERE 1=1"
    params = []
    if class_name:
        q += " AND class_name=?"; params.append(class_name)
    if date_filter:
        q += " AND date=?"; params.append(date_filter)
    q += " ORDER BY time DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Mobile usage helpers ──────────────────────────────────────────

def log_mobile_usage(student_name, class_name, date, first_seen, last_seen, total_seconds):
    """Upsert mobile usage — update if record exists for same student+class+date."""
    conn = get_conn()
    existing = conn.execute(
        "SELECT id FROM mobile_usage WHERE student_name=? AND class_name=? AND date=?",
        (student_name, class_name, date)
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE mobile_usage SET last_seen=?, total_seconds=? WHERE id=?",
            (last_seen, total_seconds, existing["id"])
        )
    else:
        conn.execute(
            "INSERT INTO mobile_usage (student_name, class_name, date, first_seen, last_seen, total_seconds) "
            "VALUES (?,?,?,?,?,?)",
            (student_name, class_name, date, first_seen, last_seen, total_seconds)
        )
    conn.commit()
    conn.close()


def get_mobile_usage(class_name=None, date_filter=None):
    conn = get_conn()
    q = "SELECT * FROM mobile_usage WHERE 1=1"
    params = []
    if class_name:
        q += " AND class_name=?"; params.append(class_name)
    if date_filter:
        q += " AND date=?"; params.append(date_filter)
    q += " ORDER BY total_seconds DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Head down helpers ─────────────────────────────────────────────

def log_head_down(student_name, class_name, date, time, duration_sec):
    conn = get_conn()
    conn.execute(
        "INSERT INTO head_down (student_name, class_name, date, time, duration_sec) "
        "VALUES (?,?,?,?,?)",
        (student_name, class_name, date, time, duration_sec)
    )
    conn.commit()
    conn.close()


def get_head_down(class_name=None, date_filter=None):
    conn = get_conn()
    q = "SELECT * FROM head_down WHERE 1=1"
    params = []
    if class_name:
        q += " AND class_name=?"; params.append(class_name)
    if date_filter:
        q += " AND date=?"; params.append(date_filter)
    q += " ORDER BY date DESC, time DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Alert helpers ─────────────────────────────────────────────────

def add_alert(alert_type, student_name, class_name, message, date, time):
    conn = get_conn()
    conn.execute(
        "INSERT INTO alerts (alert_type, student_name, class_name, message, date, time) "
        "VALUES (?,?,?,?,?,?)",
        (alert_type, student_name, class_name, message, date, time)
    )
    conn.commit()
    conn.close()


def get_recent_alerts(limit=20):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY date DESC, time DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Dashboard stats ───────────────────────────────────────────────

def get_stats(today):
    conn = get_conn()
    total_students  = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    total_classes   = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
    today_present   = conn.execute(
        "SELECT COUNT(*) FROM attendance WHERE date=?", (today,)
    ).fetchone()[0]
    total_records   = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    mobile_alerts   = conn.execute(
        "SELECT COUNT(*) FROM mobile_usage WHERE date=?", (today,)
    ).fetchone()[0]
    head_down_count = conn.execute(
        "SELECT COUNT(*) FROM head_down WHERE date=?", (today,)
    ).fetchone()[0]
    conn.close()
    return {
        "total_students":  total_students,
        "total_classes":   total_classes,
        "today_present":   today_present,
        "total_records":   total_records,
        "mobile_alerts":   mobile_alerts,
        "head_down_count": head_down_count,
    }
