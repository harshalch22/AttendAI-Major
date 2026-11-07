# ◉ AttendAI Pro — Setup Guide (Windows 11 + Python 3.12)

## INSTALL IN 4 STEPS

### Step 1 — Open terminal inside attendai_pro folder
Right-click the folder → "Open in Terminal"
OR in VS Code: File → Open Folder → select attendai_pro → Ctrl+`

### Step 2 — Create virtual environment
```
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install packages
```
pip install flask opencv-python deepface tf-keras numpy pandas Pillow ultralytics
```
⏳ Takes 5–8 minutes. Downloads AI models automatically.

### Step 4 — Run
```
python app.py
```
Open: http://localhost:5000
Login: admin / admin123

---

## HOW TO USE (in order)

1. **Classes page** → Add your classes (e.g. CS-A, CS-B)
2. **Students page** → Add students, select their class
3. **Students page** → Capture 8 face images per student
4. **Start Session** → Select a class → Live monitor opens
5. Students face camera → attendance auto-marked
6. **Reports** → Attendance / Emotions / Mobile / Head Down

---

## FEATURES

| Feature | Technology |
|---|---|
| Face Recognition | DeepFace ArcFace (highest accuracy) |
| Emotion Detection | DeepFace (7 emotions + confidence %) |
| Mobile Detection | YOLOv8 (real-time object detection) |
| Head Down Detection | MediaPipe Face Mesh / geometry |
| Database | SQLite (6 tables) |
| Web Interface | Flask + custom dark UI |

## PROJECT STRUCTURE

```
attendai_pro/
├── app.py              ← Run this
├── face_engine.py      ← All AI logic
├── database/
│   └── database.py     ← All DB operations
├── student_images/     ← Face images saved here
├── static/css/
└── templates/          ← All HTML pages
```
