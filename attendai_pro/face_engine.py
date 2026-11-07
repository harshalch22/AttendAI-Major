"""
face_engine.py  — AttendAI Pro (Speed Optimised)
-------------------------------------------------
Key speed improvements:
  1. Recognition runs every 8 frames only (not every frame)
  2. Last known name is cached and reused between recognition frames
  3. Frame resized to 640px before processing (faster detection)
  4. Emotion detected only when recognition runs (not every frame)
  5. VGG-Face model (already on disk) — no extra download needed
"""

import os
import cv2
import numpy as np
import base64
import datetime

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "student_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Optional: MediaPipe ───────────────────────────────────────────
try:
    import mediapipe as mp
    MP_FACE_MESH = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    MEDIAPIPE_OK = True
    print("[INFO] MediaPipe loaded.")
except Exception:
    MP_FACE_MESH = None
    MEDIAPIPE_OK = False

# ── Optional: YOLO ────────────────────────────────────────────────
YOLO_MODEL = None
YOLO_OK    = False

def _load_yolo():
    global YOLO_MODEL, YOLO_OK
    if YOLO_OK:
        return
    try:
        from ultralytics import YOLO
        YOLO_MODEL = YOLO("yolov8n.pt")
        YOLO_OK    = True
        print("[INFO] YOLOv8 loaded.")
    except Exception as e:
        print(f"[WARN] YOLOv8 not available: {e}")

# ── Session state ─────────────────────────────────────────────────
_marked_this_session  = set()
_mobile_start_times   = {}
_mobile_total_seconds = {}
_head_down_start      = {}

# ── Speed: frame skip + cache ─────────────────────────────────────
_frame_counter = 0
RECOG_EVERY    = 8          # recognise every N frames — increase if too slow
_face_cache    = []         # cached results from last recognition

# ── Callbacks ─────────────────────────────────────────────────────
_mark_attendance_cb = None
_log_emotion_cb     = None
_log_mobile_cb      = None
_log_head_down_cb   = None
_add_alert_cb       = None
_current_class_id   = None
_current_class_name = ""

RECOG_THRESHOLD = 0.75      # raise to 0.85 if showing Unknown too often


def set_session(class_id, class_name,
                mark_cb, emotion_cb, mobile_cb, head_down_cb, alert_cb):
    global _current_class_id, _current_class_name
    global _mark_attendance_cb, _log_emotion_cb
    global _log_mobile_cb, _log_head_down_cb, _add_alert_cb
    _current_class_id    = class_id
    _current_class_name  = class_name
    _mark_attendance_cb  = mark_cb
    _log_emotion_cb      = emotion_cb
    _log_mobile_cb       = mobile_cb
    _log_head_down_cb    = head_down_cb
    _add_alert_cb        = alert_cb
    _load_yolo()


def reset_session():
    global _frame_counter, _face_cache
    _marked_this_session.clear()
    _mobile_start_times.clear()
    _mobile_total_seconds.clear()
    _head_down_start.clear()
    _frame_counter = 0
    _face_cache    = []


# ── Helpers ───────────────────────────────────────────────────────

def _now_str():
    n = datetime.datetime.now()
    return n.strftime("%Y-%m-%d"), n.strftime("%H:%M:%S")


def _decode_b64(b64: str):
    if "," in b64:
        b64 = b64.split(",")[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _resize_frame(frame, max_w=640):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame, 1.0
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale))), scale


def _detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
    return faces if len(faces) > 0 else []


# ── Registration ──────────────────────────────────────────────────

def save_face(student_name: str, image_b64: str) -> dict:
    frame = _decode_b64(image_b64)
    if frame is None:
        return {"ok": False, "message": "Cannot decode image.", "count": 0}

    faces = _detect_faces(frame)
    if len(faces) == 0:
        return {"ok": False, "message": "No face detected. Move closer and ensure good lighting.", "count": 0}

    folder   = os.path.join(IMAGES_DIR, student_name)
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.endswith(".jpg")])

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = 30
    x1  = max(0, x - pad);              y1 = max(0, y - pad)
    x2  = min(frame.shape[1], x+w+pad); y2 = min(frame.shape[0], y+h+pad)
    cv2.imwrite(os.path.join(folder, f"{student_name}_{existing+1}.jpg"), frame[y1:y2, x1:x2])

    count = existing + 1
    msg   = f"Image {count} saved!"
    if count >= 8:
        msg += " All 8 done — high accuracy ready!"
    return {"ok": True, "message": msg, "count": count}


def get_face_count(student_name: str) -> int:
    folder = os.path.join(IMAGES_DIR, student_name)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


# ── Face Recognition ──────────────────────────────────────────────

def recognise_face(face_crop, student_names: list) -> tuple:
    """Returns (name, confidence%) using VGG-Face (already on disk)."""
    if not student_names or face_crop is None or face_crop.size == 0:
        return "Unknown", 0.0
    try:
        from deepface import DeepFace
        tmp = os.path.join(IMAGES_DIR, "_query.jpg")
        cv2.imwrite(tmp, face_crop)

        best_name = "Unknown"
        best_dist = 1.0

        for name in student_names:
            folder = os.path.join(IMAGES_DIR, name)
            if not os.path.isdir(folder):
                continue
            imgs = [os.path.join(folder, f)
                    for f in os.listdir(folder) if f.endswith(".jpg")]
            if not imgs:
                continue

            dists = []
            for img_path in imgs:
                try:
                    r = DeepFace.verify(
                        img1_path=tmp, img2_path=img_path,
                        model_name="VGG-Face",
                        detector_backend="opencv",
                        enforce_detection=False,
                        silent=True
                    )
                    dists.append(r.get("distance", 1.0))
                except Exception:
                    continue

            if dists:
                avg = float(np.mean(sorted(dists)[:3]))
                if avg < best_dist:
                    best_dist = avg
                    if avg < RECOG_THRESHOLD:
                        best_name = name

        if os.path.exists(tmp):
            os.remove(tmp)

        conf = max(0.0, round((1 - best_dist / RECOG_THRESHOLD) * 100, 1))
        return best_name, conf

    except Exception as e:
        print(f"[Recognition Error] {e}")
        return "Unknown", 0.0


# ── Emotion ───────────────────────────────────────────────────────

EMOTION_EMOJI = {"happy":"😊","neutral":"😐","sad":"😢",
                 "angry":"😠","surprise":"😲","fear":"😨","disgust":"🤢"}

def detect_emotion(face_crop) -> tuple:
    if face_crop is None or face_crop.size == 0:
        return "Neutral", 0.0
    try:
        from deepface import DeepFace
        r = DeepFace.analyze(face_crop, actions=["emotion"],
                             enforce_detection=False, silent=True)
        if isinstance(r, list): r = r[0]
        emo  = r.get("dominant_emotion", "neutral").capitalize()
        conf = round(float(r.get("emotion", {}).get(emo.lower(), 0)), 1)
        return emo, conf
    except Exception:
        return _heuristic_emotion(face_crop)


def _heuristic_emotion(img) -> tuple:
    try:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w  = gray.shape
        mouth = gray[int(h*.6):h, int(w*.2):int(w*.8)]
        diff  = float(np.mean(mouth)) - float(np.mean(gray)) if mouth.size else 0
        if diff > 8:  return "Happy",   65.0
        if diff < -8: return "Sad",     65.0
        return "Neutral", 65.0
    except Exception:
        return "Neutral", 0.0


# ── Head Down ─────────────────────────────────────────────────────

def check_head_pose(frame, face_rect) -> bool:
    if MEDIAPIPE_OK and MP_FACE_MESH:
        try:
            x, y, w, h = face_rect
            roi     = frame[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
            results = MP_FACE_MESH.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                return abs(lms[1].y - lms[152].y) < abs(lms[1].y - lms[10].y) * 0.7
        except Exception:
            pass
    x, y, w, h = face_rect
    return (y + h // 2) > frame.shape[0] * 0.65 and h / max(w, 1) < 0.85


# ── Mobile Detection ──────────────────────────────────────────────

def detect_mobile(frame) -> list:
    if not YOLO_OK or YOLO_MODEL is None:
        return []
    try:
        results = YOLO_MODEL(frame, verbose=False, conf=0.45)
        return [tuple(map(int, b.xyxy[0]))
                for r in results for b in r.boxes if int(b.cls[0]) == 67]
    except Exception:
        return []


# ── Main Frame Processor ──────────────────────────────────────────

def process_frame(frame, student_names: list) -> tuple:
    global _frame_counter, _face_cache
    date_str, time_str = _now_str()
    H, W = frame.shape[:2]

    # Resize for speed, run face detection on smaller image
    small, scale    = _resize_frame(frame, 640)
    mobile_boxes    = detect_mobile(small)
    mobile_detected = len(mobile_boxes) > 0

    # Draw mobile alerts
    for (mx1, my1, mx2, my2) in mobile_boxes:
        ox1,oy1 = int(mx1/scale), int(my1/scale)
        ox2,oy2 = int(mx2/scale), int(my2/scale)
        cv2.rectangle(frame, (ox1,oy1), (ox2,oy2), (0,50,220), 3)
        cv2.putText(frame, "MOBILE DETECTED", (ox1, oy1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,50,220), 2)

    # Detect faces on small image, scale back
    faces_small = _detect_faces(small)
    faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale))
             for (x,y,w,h) in faces_small]

    _frame_counter += 1
    should_recognise = (_frame_counter % RECOG_EVERY == 0) or \
                       (len(_face_cache) == 0 and len(faces) > 0)

    # ── Run recognition (every RECOG_EVERY frames only) ───────────
    if should_recognise and len(faces) > 0:
        new_cache = []
        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            name, rc  = recognise_face(crop, student_names)
            emo,  ec  = detect_emotion(crop)
            hdown     = check_head_pose(frame, (x,y,w,h))
            new_cache.append({"name":name,"rc":rc,"emo":emo,"ec":ec,
                               "hdown":hdown,"box":(x,y,w,h)})

            if name != "Unknown":
                # Mark attendance
                if name not in _marked_this_session and _mark_attendance_cb:
                    if _mark_attendance_cb(name, _current_class_id,
                                           _current_class_name, date_str, time_str):
                        _marked_this_session.add(name)
                        if _log_emotion_cb:
                            _log_emotion_cb(name, _current_class_name,
                                            emo, ec, date_str, time_str)
                # Mobile usage
                if mobile_detected and _log_mobile_cb:
                    if name not in _mobile_start_times:
                        _mobile_start_times[name] = datetime.datetime.now()
                        if _add_alert_cb:
                            _add_alert_cb("MOBILE", name, _current_class_name,
                                          f"{name} is using mobile!", date_str, time_str)
                    else:
                        secs = int((datetime.datetime.now()-_mobile_start_times[name]).total_seconds())
                        _log_mobile_cb(name, _current_class_name, date_str,
                                       _mobile_start_times[name].strftime("%H:%M:%S"),
                                       time_str, secs)
                elif name in _mobile_start_times:
                    del _mobile_start_times[name]

                # Head down
                if hdown and _log_head_down_cb:
                    if name not in _head_down_start:
                        _head_down_start[name] = datetime.datetime.now()
                        if _add_alert_cb:
                            _add_alert_cb("HEAD_DOWN", name, _current_class_name,
                                          f"{name} looking down!", date_str, time_str)
                    else:
                        dur = int((datetime.datetime.now()-_head_down_start[name]).total_seconds())
                        if dur >= 5:
                            _log_head_down_cb(name, _current_class_name,
                                              date_str, time_str, dur)
                elif name in _head_down_start:
                    del _head_down_start[name]

        _face_cache = new_cache

    # ── Draw cached annotations every frame (smooth video) ────────
    for e in _face_cache:
        x,y,w,h = e["box"]
        colour   = (0,210,90) if e["name"] != "Unknown" else (0,60,210)
        cv2.rectangle(frame, (x,y), (x+w,y+h), colour, 2)
        if e["hdown"]:
            cv2.putText(frame,"HEAD DOWN",(x,y-28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
        lbl1 = f"{e['name']} ({e['rc']:.0f}%)"
        lbl2 = f"{EMOTION_EMOJI.get(e['emo'].lower(),'')} {e['emo']} ({e['ec']:.0f}%)"
        cv2.rectangle(frame,(x,y+h),(x+w,y+h+42),colour,cv2.FILLED)
        cv2.putText(frame,lbl1,(x+4,y+h+16),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1)
        cv2.putText(frame,lbl2,(x+4,y+h+34),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)

    # Status bar
    cv2.rectangle(frame,(0,0),(W,30),(15,15,15),cv2.FILLED)
    cv2.putText(frame,f"{date_str} {time_str} | {_current_class_name}",(8,21),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,220,140),2)

    if mobile_detected:
        cv2.rectangle(frame,(0,H-36),(W,H),(0,0,180),cv2.FILLED)
        cv2.putText(frame,"MOBILE PHONE DETECTED",(10,H-12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)

    return frame, _face_cache


# ── MJPEG Generator ───────────────────────────────────────────────

def generate_frames(student_names: list):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        blank = np.zeros((360,640,3),dtype=np.uint8)
        cv2.putText(blank,"Camera not found",(80,180),cv2.FONT_HERSHEY_SIMPLEX,1,(0,60,220),2)
        _,buf = cv2.imencode(".jpg",blank)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        return

    print(f"[INFO] Camera open. Recognising every {RECOG_EVERY} frames.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, _ = process_frame(frame, student_names)
            _, buf   = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
    finally:
        cap.release()
        print("[INFO] Camera released.")
