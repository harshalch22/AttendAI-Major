"""
app.py — AttendAI Pro
---------------------
All Flask routes for the complete system.
"""

import os
import sys
import datetime
from flask import (Flask, render_template, request, redirect,
                   url_for, session, Response, flash, jsonify)

sys.path.insert(0, os.path.dirname(__file__))

from database.database import (
    init_db,
    add_class, get_all_classes, get_class_by_id, delete_class,
    add_student, get_all_students, get_students_by_class,
    get_student_names_by_class, student_exists, delete_student,
    mark_attendance, get_attendance, get_attendance_dates,
    log_emotion, get_emotions,
    log_mobile_usage, get_mobile_usage,
    log_head_down, get_head_down,
    add_alert, get_recent_alerts,
    get_stats
)
from face_engine import (
    save_face, get_face_count,
    set_session, reset_session,
    generate_frames
)

app = Flask(__name__)
app.secret_key = "attendai_pro_2024_secret"

init_db()

USERNAME = "admin"
PASSWORD = "admin123"

# Currently active session class (stored server-side)
_active_class = {"id": None, "name": ""}


def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped


# ─────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        if (request.form.get("username") == USERNAME and
                request.form.get("password") == PASSWORD):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Wrong credentials. Use: admin / admin123"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    today   = datetime.date.today().isoformat()
    stats   = get_stats(today)
    alerts  = get_recent_alerts(10)
    classes = get_all_classes()
    return render_template("dashboard.html",
                           stats=stats, alerts=alerts,
                           classes=classes, today=today)


# ─────────────────────────────────────────────────────────────────
# Classes (Divisions)
# ─────────────────────────────────────────────────────────────────

@app.route("/classes", methods=["GET", "POST"])
@login_required
def classes():
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        dept  = request.form.get("department", "").strip()
        year  = request.form.get("year", "").strip()
        if not name or not dept or not year:
            flash("All fields are required.", "danger")
        elif add_class(name, dept, year):
            flash(f"Class '{name}' created successfully!", "success")
        else:
            flash(f"Class '{name}' already exists.", "warning")
        return redirect(url_for("classes"))

    all_classes = get_all_classes()
    # Add student count per class
    for c in all_classes:
        c["student_count"] = len(get_students_by_class(c["id"]))
    return render_template("classes.html", classes=all_classes)


@app.route("/classes/delete/<int:class_id>")
@login_required
def delete_class_route(class_id):
    delete_class(class_id)
    flash("Class deleted.", "warning")
    return redirect(url_for("classes"))


# ─────────────────────────────────────────────────────────────────
# Students
# ─────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    classes = get_all_classes()

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        roll_no  = request.form.get("roll_no", "").strip()
        class_id = request.form.get("class_id", "").strip()

        if not name or not roll_no or not class_id:
            flash("All fields are required.", "danger")
        elif add_student(name, roll_no, int(class_id)):
            flash(f"✅ Student '{name}' added! Now capture their face.", "success")
        else:
            flash(f"Roll number '{roll_no}' already exists.", "warning")
        return redirect(url_for("register"))

    # Filter by class if requested
    filter_class = request.args.get("class_id")
    if filter_class:
        students = get_students_by_class(int(filter_class))
    else:
        students = get_all_students()

    for s in students:
        s["face_count"] = get_face_count(s["name"])

    return render_template("register.html",
                           students=students, classes=classes,
                           filter_class=filter_class)


@app.route("/students/delete/<int:student_id>")
@login_required
def delete_student_route(student_id):
    delete_student(student_id)
    flash("Student deleted.", "warning")
    return redirect(url_for("register"))


@app.route("/capture_face", methods=["POST"])
@login_required
def capture_face():
    data  = request.get_json()
    name  = data.get("name", "").strip()
    image = data.get("image", "")
    if not name or not student_exists(name):
        return jsonify({"ok": False, "message": "Student not found."})
    result = save_face(name, image)
    return jsonify(result)


# ─────────────────────────────────────────────────────────────────
# Attendance Session
# ─────────────────────────────────────────────────────────────────

@app.route("/select_class")
@login_required
def select_class():
    classes = get_all_classes()
    return render_template("select_class.html", classes=classes)


@app.route("/start/<int:class_id>")
@login_required
def start(class_id):
    cls = get_class_by_id(class_id)
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for("select_class"))

    _active_class["id"]   = class_id
    _active_class["name"] = cls["name"]

    reset_session()
    set_session(
        class_id       = class_id,
        class_name     = cls["name"],
        mark_cb        = mark_attendance,
        emotion_cb     = log_emotion,
        mobile_cb      = log_mobile_usage,
        head_down_cb   = log_head_down,
        alert_cb       = add_alert,
    )
    return redirect(url_for("monitor"))


@app.route("/monitor")
@login_required
def monitor():
    if not _active_class["id"]:
        return redirect(url_for("select_class"))
    return render_template("monitor.html",
                           class_name=_active_class["name"])


@app.route("/video_feed")
@login_required
def video_feed():
    class_id = _active_class.get("id")
    if class_id:
        names = get_student_names_by_class(class_id)
    else:
        names = []
    return Response(
        generate_frames(names),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────

@app.route("/attendance")
@login_required
def attendance():
    today       = datetime.date.today().isoformat()
    date_filter = request.args.get("date", today)
    class_id    = request.args.get("class_id", "")
    records     = get_attendance(
        class_id    = int(class_id) if class_id else None,
        date_filter = date_filter
    )
    classes     = get_all_classes()
    all_dates   = get_attendance_dates()
    return render_template("attendance.html",
                           records=records, classes=classes,
                           all_dates=all_dates[:15],
                           date=date_filter, class_id=class_id)


@app.route("/emotions")
@login_required
def emotions():
    today       = datetime.date.today().isoformat()
    date_filter = request.args.get("date", today)
    class_name  = request.args.get("class_name", "")
    records     = get_emotions(
        class_name  = class_name or None,
        date_filter = date_filter
    )
    classes = get_all_classes()
    return render_template("emotions.html",
                           records=records, classes=classes,
                           date=date_filter, class_name=class_name)


@app.route("/mobile_report")
@login_required
def mobile_report():
    today       = datetime.date.today().isoformat()
    date_filter = request.args.get("date", today)
    class_name  = request.args.get("class_name", "")
    records     = get_mobile_usage(
        class_name  = class_name or None,
        date_filter = date_filter
    )
    classes = get_all_classes()
    return render_template("mobile_report.html",
                           records=records, classes=classes,
                           date=date_filter, class_name=class_name)


@app.route("/head_down_report")
@login_required
def head_down_report():
    today       = datetime.date.today().isoformat()
    date_filter = request.args.get("date", today)
    class_name  = request.args.get("class_name", "")
    records     = get_head_down(
        class_name  = class_name or None,
        date_filter = date_filter
    )
    classes = get_all_classes()
    return render_template("head_down_report.html",
                           records=records, classes=classes,
                           date=date_filter, class_name=class_name)


@app.route("/alerts")
@login_required
def alerts():
    records = get_recent_alerts(50)
    return render_template("alerts.html", records=records)


# ─────────────────────────────────────────────────────────────────
# API — live alert polling (used by monitor page)
# ─────────────────────────────────────────────────────────────────

@app.route("/api/alerts")
@login_required
def api_alerts():
    return jsonify(get_recent_alerts(5))


# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  AttendAI Pro — Starting...")
    print("  Open: http://localhost:5000")
    print("  Login: admin / admin123")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=5000)
