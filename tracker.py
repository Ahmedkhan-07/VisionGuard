"""
╔══════════════════════════════════════════════════════════╗
║           SMART PRESENCE TRACKER  v2.0                   ║
║   ID Login → Face Capture → Live Track → Email Alert     ║
╚══════════════════════════════════════════════════════════╝

INSTALL:
  pip install opencv-python face_recognition numpy pandas

RUN:
  python tracker.py

KEYBOARD CONTROLS:
  Type ID  + ENTER  →  Login
  BACKSPACE         →  Delete typed character
  E                 →  End session (confirmation screen)
  ESC               →  Back to login  (clears face data)
  Q                 →  Quit immediately
"""

import cv2
import face_recognition
import numpy as np
import pandas as pd
import smtplib
import threading
import time
import csv
import sys
import atexit
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from datetime             import datetime
import mediapipe as mp

# ── Load config ────────────────────────────────────────────
try:
    import config
except ImportError:
    print("[ERROR] config.py not found. Please create it first.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════
#  COLORS  (BGR)
# ══════════════════════════════════════════════════════════
CLR_GREEN  = (0,  210,  80)
CLR_RED    = (30,  50, 220)
CLR_YELLOW = (20, 200, 230)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,    0,   0)
CLR_DARK   = (18,  18,  25)
CLR_GOLD   = (255, 175,   0)
CLR_GRAY   = (140, 140, 140)

# ══════════════════════════════════════════════════════════
#  STATES
# ══════════════════════════════════════════════════════════
S_LOGIN    = 0
S_CAPTURE  = 1
S_TRACKING = 2
S_ENDING   = 3

# ══════════════════════════════════════════════════════════
#  PRIVACY — face data lives ONLY in RAM, never on disk
# ══════════════════════════════════════════════════════════
_ram_encodings = []   # list of face encoding arrays
_ram_photos    = []   # list of numpy BGR frames (never saved)

def _clear_face_data():
    """Wipe all face data from RAM. Called on exit and session end."""
    global _ram_encodings, _ram_photos
    _ram_encodings = []
    _ram_photos    = []
    print("[PRIVACY] Face data cleared from memory.")

atexit.register(_clear_face_data)   # runs even on crash / Ctrl-C

# ══════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ══════════════════════════════════════════════════════════
def load_people(path="people.csv"):
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            df[col] = df[col].str.strip()
        return df
    except FileNotFoundError:
        print(f"[ERROR] {path} not found.")
        return pd.DataFrame(columns=["ID","Name","Department",
                                     "Authority_Name","Authority_Email"])
    except Exception as e:
        print(f"[ERROR] Cannot read {path}: {e}")
        return pd.DataFrame()

def find_person(uid, df):
    if df.empty:
        return None
    match = df[df["ID"] == uid.strip().upper()]
    return match.iloc[0].to_dict() if not match.empty else None

# ══════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════
def log_event(pid, name, event, detail=""):
    path   = "presence_log.csv"
    exists = False
    try:
        open(path, "r").close()
        exists = True
    except FileNotFoundError:
        pass
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp","ID","Name","Event","Detail"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    pid, name, event, detail])

# ══════════════════════════════════════════════════════════
#  EMAIL ALERT
# ══════════════════════════════════════════════════════════
_email_lock = threading.Lock()

def send_alert_email(person, missing_since, snapshot_frame):
    """Send alert in a background thread so UI never freezes."""
    def _worker():
        with _email_lock:
            try:
                duration = int(time.time() - missing_since)
                miss_str = datetime.fromtimestamp(missing_since).strftime("%H:%M:%S")
                now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                msg             = MIMEMultipart()
                msg["From"]     = config.SENDER_EMAIL
                msg["To"]       = person["Authority_Email"]
                msg["Subject"]  = (f"\u26a0\ufe0f PRESENCE ALERT \u2014 "
                                   f"{person['Name']} ({person['ID']}) Missing")

                body = (
                    f"Dear {person['Authority_Name']},\n\n"
                    f"This is to inform you that the following faculty member is NOT PRESENT at their designated location.\n\n"
                    f"{'─'*42}\n"
                    f"  Name        : {person['Name']}\n"
                    f"  ID          : {person['ID']}\n"
                    f"  Department  : {person['Department']}\n"
                    f"{'─'*42}\n"
                    f"  Last seen   : {miss_str}\n"
                    f"  Missing for : {duration} seconds\n"
                    f"  Alert time  : {now_str}\n"
                    f"{'─'*42}\n\n"
                    f"A camera snapshot is attached.\n"
                    f"Please verify the situation immediately.\n\n"
                    f"\u2014 Smart Presence Tracker (Automated System)"
                )
                msg.attach(MIMEText(body, "plain"))

                # Attach snapshot
                if snapshot_frame is not None:
                    ok, buf = cv2.imencode(".jpg", snapshot_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        img = MIMEImage(buf.tobytes(), name="alert_snapshot.jpg")
                        msg.attach(img)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
                    srv.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                    srv.send_message(msg)

                print(f"[EMAIL] Alert sent to {person['Authority_Email']}")
                log_event(person["ID"], person["Name"], "ALERT_EMAIL_SENT",
                          f"missing {duration}s → {person['Authority_Email']}")

            except Exception as e:
                print(f"[EMAIL ERROR] {e}")

    threading.Thread(target=_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════
#  DRAWING UTILITIES
# ══════════════════════════════════════════════════════════
_FONT_D = cv2.FONT_HERSHEY_DUPLEX
_FONT_S = cv2.FONT_HERSHEY_SIMPLEX

def blend_rect(frame, x1, y1, x2, y2, alpha=0.75, color=None):
    """Draw a semi-transparent filled rectangle with a border."""
    if color is None:
        color = CLR_DARK
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi  = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_GOLD, 1)

def draw_text(frame, text, pos, scale=0.6, color=CLR_WHITE,
              thick=1, shadow=True, font=None):
    f = font if font else _FONT_D
    if shadow:
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    f, scale, CLR_BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, f, scale, color, thick, cv2.LINE_AA)

def draw_bar(frame, x, y, w, h, pct, fg=CLR_GREEN, bg=(40, 40, 40)):
    pct = max(0.0, min(1.0, pct))
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    filled = int(w * pct)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), fg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), CLR_GRAY, 1)

# ══════════════════════════════════════════════════════════
#  HOG PERSON DETECTOR (fallback when face not visible)
# ══════════════════════════════════════════════════════════
# ── MediaPipe Pose — new API (0.10.30+) ───────────────────
_mp_base_options   = mp.tasks.BaseOptions
_mp_pose_landmarker = mp.tasks.vision.PoseLandmarker
_mp_pose_options   = mp.tasks.vision.PoseLandmarkerOptions
_mp_running_mode   = mp.tasks.vision.RunningMode

import urllib.request, os
_MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(_MODEL_PATH):
    print("[MEDIAPIPE] Downloading pose model (~5MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        _MODEL_PATH
    )
    print("[MEDIAPIPE] Model downloaded!")

_pose_options = _mp_pose_options(
    base_options=_mp_base_options(model_asset_path=_MODEL_PATH),
    running_mode=_mp_running_mode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_pose_landmarker_instance = _mp_pose_landmarker.create_from_options(_pose_options)

def person_in_frame_mediapipe(frame):
    """Returns True if MediaPipe detects a human body/skeleton in frame."""
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result    = _pose_landmarker_instance.detect(mp_image)
    if not result.pose_landmarks:
        return False
    # At least 5 landmarks must have high visibility
    lms     = result.pose_landmarks[0]
    visible = sum(1 for lm in lms if lm.visibility > 0.5)
    return visible >= 5

# ══════════════════════════════════════════════════════════
#  SCREEN RENDERERS
# ══════════════════════════════════════════════════════════

def render_login(frame, typed_id, error_msg, df):
    fh, fw = frame.shape[:2]

    # Dim background
    overlay = np.full_like(frame, CLR_DARK, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    draw_text(frame, "SMART PRESENCE TRACKER",
              (fw // 2 - 285, 75), scale=1.1, color=CLR_GOLD, thick=2)
    draw_text(frame, "Face Recognition  |  Auto Alert  |  Privacy Safe",
              (fw // 2 - 260, 108), scale=0.5, color=CLR_GRAY)

    # Input box
    bx, by, bw, bh = fw // 2 - 240, fh // 2 - 95, 480, 145
    blend_rect(frame, bx, by, bx + bw, by + bh, alpha=0.85)

    draw_text(frame, "Enter Your Employee / Faculty ID:",
              (bx + 20, by + 38), scale=0.58, color=CLR_GRAY)

    display = (typed_id + "|") if typed_id else "_ _ _ _ _ _ _"
    col     = CLR_YELLOW if typed_id else CLR_GRAY
    draw_text(frame, display, (bx + 20, by + 100),
              scale=1.1, color=col, thick=2)

    if error_msg:
        draw_text(frame, f"  {error_msg}",
                  (fw // 2 - 190, fh // 2 + 80), scale=0.62, color=CLR_RED)

    # Controls hint
    draw_text(frame, "ENTER = Confirm      BACKSPACE = Delete      Q = Quit",
              (fw // 2 - 265, fh - 45), scale=0.48, color=CLR_GRAY)

    # Sample IDs
    if not df.empty:
        draw_text(frame, "Registered IDs:", (30, fh - 135),
                  scale=0.48, color=CLR_GRAY)
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            draw_text(frame,
                      f"  {row['ID']}  |  {row['Name']}  ({row['Department']})",
                      (30, fh - 115 + i * 22), scale=0.42, color=CLR_GRAY)


def render_capture(frame, person, phase, countdown, total=3):
    fh, fw = frame.shape[:2]

    # Top bar
    blend_rect(frame, 0, 0, fw, 72, alpha=0.85)
    draw_text(frame,
              f"FACE REGISTRATION  —  {person['Name']}  |  {person['ID']}",
              (18, 44), scale=0.75, color=CLR_GOLD, thick=2)

    # Face guide oval
    cv2.ellipse(frame, (fw // 2, fh // 2), (130, 170),
                0, 0, 360, CLR_GOLD, 2)
    draw_text(frame, "Align face inside oval",
              (fw // 2 - 110, fh // 2 + 190),
              scale=0.55, color=CLR_GOLD)

    # Bottom bar
    blend_rect(frame, 0, fh - 135, fw, fh, alpha=0.85)

    # Progress dots
    dot_y = fh - 65
    for i in range(total):
        cx  = fw // 2 - (total * 42) // 2 + i * 42 + 21
        col = CLR_GREEN if i < phase else \
              (CLR_YELLOW if i == phase else (50, 50, 50))
        cv2.circle(frame, (cx, dot_y), 14, col, -1)
        cv2.circle(frame, (cx, dot_y), 15, CLR_WHITE, 1)
        draw_text(frame, str(i + 1), (cx - 6, dot_y + 5),
                  scale=0.45, color=CLR_BLACK, shadow=False)

    draw_text(frame, f"Photo {phase + 1} of {total}",
              (fw // 2 - 65, fh - 90), scale=0.6, color=CLR_WHITE)

    instructions = [
        "Step 1:  Look STRAIGHT at the camera",
        "Step 2:  Turn your face slightly to the LEFT",
        "Step 3:  Turn your face slightly to the RIGHT",
    ]
    if phase < len(instructions):
        draw_text(frame, instructions[phase],
                  (fw // 2 - 210, fh - 108), scale=0.58, color=CLR_YELLOW)

    if countdown > 0:
        draw_text(frame, f"Capturing in  {countdown}s ...",
                  (fw // 2 - 125, fh - 22), scale=0.65, color=CLR_YELLOW)
        cv2.putText(frame, str(countdown),
                    (fw // 2 - 35, fh // 2 + 40),
                    _FONT_D, 3.5, CLR_GOLD, 5, cv2.LINE_AA)
    else:
        draw_text(frame, "  Smile!  Capturing ...",
                  (fw // 2 - 105, fh - 22), scale=0.65, color=CLR_GREEN)

    # No-face warning shown by caller via an extra draw call


def render_tracking(frame, person, is_present, body_only,
                    missing_since, session_start, face_locs,
                    alert_count, show_alert_flash, flash_until):
    fh, fw  = frame.shape[:2]
    now     = time.time()

    # ── Face bounding boxes ────────────────────────────────
    for (top, right, bottom, left) in face_locs:
        col = CLR_GREEN if is_present else CLR_YELLOW
        cv2.rectangle(frame, (left, top), (right, bottom), col, 2)
        cv2.rectangle(frame, (left, bottom - 28), (right, bottom), col, -1)
        label = person["Name"] if is_present else "Tracking..."
        draw_text(frame, label, (left + 4, bottom - 8),
                  scale=0.48, color=CLR_BLACK, shadow=False)

    # ── Top header bar ────────────────────────────────────
    blend_rect(frame, 0, 0, fw, 62, alpha=0.82)

    if is_present:
        status_txt, status_col = "  PRESENT", CLR_GREEN
    elif body_only:
        status_txt, status_col = "  BACK TURNED", CLR_YELLOW
    else:
        status_txt, status_col = "  MISSING", CLR_RED

    draw_text(frame, status_txt, (12, 42),
              scale=0.92, color=status_col, thick=2)
    draw_text(frame,
              f"{person['Name']}  |  {person['ID']}  |  {person['Department']}",
              (215, 42), scale=0.58, color=CLR_WHITE)

    # Session timer (top-right)
    elapsed  = int(now - session_start)
    mm, ss   = divmod(elapsed, 60)
    hh, mm2  = divmod(mm, 60)
    timer_str = f"{hh:02d}:{mm2:02d}:{ss:02d}"
    draw_text(frame, f"Session: {timer_str}", (fw - 200, 42),
              scale=0.58, color=CLR_GRAY)

    # ── Bottom-left status panel ──────────────────────────
    pw, ph = 320, 148
    blend_rect(frame, 0, fh - ph, pw, fh, alpha=0.82)

    if not is_present and not body_only and missing_since is not None:
        miss_dur  = now - missing_since
        remaining = max(0.0, config.MISSING_THRESHOLD_SEC - miss_dur)
        pct       = min(miss_dur / config.MISSING_THRESHOLD_SEC, 1.0)

        draw_text(frame, "  TARGET NOT DETECTED",
                  (10, fh - ph + 28), scale=0.58, color=CLR_RED)
        draw_text(frame, f"Missing  :  {int(miss_dur)}s",
                  (10, fh - ph + 56), scale=0.55, color=CLR_YELLOW)
        draw_text(frame, f"Alert in :  {int(remaining)}s",
                  (10, fh - ph + 82), scale=0.55, color=CLR_YELLOW)
        draw_bar(frame, 10, fh - ph + 100, pw - 20, 18, pct, fg=CLR_RED)

    elif body_only:
        draw_text(frame, "  Back turned / face not visible",
                  (10, fh - ph + 28), scale=0.52, color=CLR_YELLOW)
        draw_text(frame, "Body detected - not counted as missing",
                  (10, fh - ph + 56), scale=0.46, color=CLR_GRAY)
        draw_text(frame, "Waiting for face ...",
                  (10, fh - ph + 82), scale=0.46, color=CLR_GRAY)
    else:
        draw_text(frame, "  TRACKING ACTIVE",
                  (10, fh - ph + 28), scale=0.58, color=CLR_GREEN)
        draw_text(frame, "Face recognition running",
                  (10, fh - ph + 56), scale=0.48, color=CLR_GRAY)

    if alert_count > 0:
        draw_text(frame, f"Alerts sent: {alert_count}",
                  (10, fh - ph + 112), scale=0.5, color=CLR_YELLOW)

    # ── Bottom-right authority panel ──────────────────────
    aw = 370
    blend_rect(frame, fw - aw, fh - 90, fw, fh, alpha=0.82)
    draw_text(frame, "Alert Authority:",
              (fw - aw + 10, fh - 65), scale=0.48, color=CLR_GOLD)
    draw_text(frame, person["Authority_Name"],
              (fw - aw + 10, fh - 42), scale=0.52, color=CLR_WHITE)
    draw_text(frame, person["Authority_Email"],
              (fw - aw + 10, fh - 18), scale=0.45, color=CLR_GRAY)

    # ── Controls hint ─────────────────────────────────────
    draw_text(frame, "E = End Session      ESC = Back to Login      Q = Quit",
              (fw // 2 - 245, fh - 8), scale=0.42, color=CLR_GRAY)

    # ── Alert flash overlay ───────────────────────────────
    if show_alert_flash and now < flash_until:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, fh), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        draw_text(frame, "  EMAIL ALERT SENT!",
                  (fw // 2 - 215, fh // 2 - 20),
                  scale=1.4, color=CLR_WHITE, thick=3)
        draw_text(frame, f"Notified: {person['Authority_Name']}",
                  (fw // 2 - 185, fh // 2 + 42),
                  scale=0.72, color=CLR_YELLOW)


def render_ending(frame, person, session_start, alert_count):
    fh, fw = frame.shape[:2]

    overlay = np.full_like(frame, CLR_DARK, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    bw, bh = 520, 385
    bx     = fw // 2 - bw // 2
    by     = fh // 2 - bh // 2
    blend_rect(frame, bx, by, bx + bw, by + bh, alpha=0.92)

    draw_text(frame, "END SESSION",
              (bx + bw // 2 - 110, by + 50),
              scale=1.0, color=CLR_GOLD, thick=2)

    elapsed      = int(time.time() - session_start)
    mm, ss       = divmod(elapsed, 60)
    hh, mm2      = divmod(mm, 60)
    duration_str = f"{hh:02d}:{mm2:02d}:{ss:02d}"

    rows = [
        ("Name",       person["Name"]),
        ("ID",         person["ID"]),
        ("Department", person["Department"]),
        ("Duration",   duration_str),
        ("Alerts Sent",str(alert_count)),
    ]
    for i, (label, val) in enumerate(rows):
        y = by + 98 + i * 34
        draw_text(frame, f"{label:<12}:  {val}", (bx + 30, y),
                  scale=0.60, color=CLR_WHITE)

    # Privacy notice box
    blend_rect(frame, bx + 12, by + 282, bx + bw - 12, by + 340,
               alpha=0.65, color=(0, 50, 0))
    draw_text(frame, "  All face photos will be deleted from memory",
              (bx + 22, by + 305), scale=0.5, color=CLR_GREEN)
    draw_text(frame, "  Session will be saved to presence_log.csv",
              (bx + 22, by + 328), scale=0.5, color=CLR_GREEN)

    draw_text(frame, "ENTER = Confirm End      ESC = Resume Session",
              (bx + bw // 2 - 220, by + bh - 22),
              scale=0.55, color=CLR_YELLOW)


# ══════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════
def main():
    global _ram_encodings, _ram_photos

    print("\n╔══════════════════════════════════════════╗")
    print("║   Smart Presence Tracker — Starting ...  ║")
    print("╚══════════════════════════════════════════╝\n")

    df = load_people("people.csv")
    if df.empty:
        print("[WARNING] people.csv is empty or missing. Add entries first.")

    # ── Open webcam ───────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Resolution: {fw} x {fh}")

    cv2.namedWindow("Smart Presence Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Presence Tracker", fw, fh)

    # ── Application state variables ───────────────────────
    state    = S_LOGIN
    typed_id = ""
    id_error = ""
    person   = None

    # Capture phase
    cap_phase     = 0      # 0, 1, 2
    cap_cd_start  = 0.0   # countdown start time
    cap_cd_secs   = 3      # countdown duration in seconds
    show_no_face  = False  # show "no face found" warning

    # Tracking state
    is_present      = False
    body_only       = False
    missing_since   = None
    session_start   = 0.0
    alert_count     = 0
    alert_flash     = False
    flash_until     = 0.0
    last_alert_time = 0.0
    face_locs       = []
    frame_idx       = 0
    body_hits       = 0        # consecutive MediaPipe body detections
    BODY_CONFIRM    = 3        # need N consecutive hits to confirm body present
    FR_EVERY        = 3        # run face recog every N frames
    MP_EVERY        = 2        # run mediapipe every N frames

    print("[READY] Camera active. Type your ID to begin.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame    = cv2.flip(frame, 1)          # mirror view
        key      = cv2.waitKey(1) & 0xFF
        now      = time.time()
        frame_idx += 1

        # ── Q always quits ─────────────────────────────────
        if key in (ord('q'), ord('Q')):
            if person:
                log_event(person["ID"], person["Name"], "SESSION_QUIT",
                          f"after {int(now - session_start)}s")
            break

        # ══════════════════════════════════════════════════
        #  LOGIN SCREEN
        # ══════════════════════════════════════════════════
        if state == S_LOGIN:
            render_login(frame, typed_id, id_error, df)

            if key in (13, 10):                        # ENTER
                if not typed_id:
                    id_error = "Please enter an ID first."
                else:
                    result = find_person(typed_id, df)
                    if result is None:
                        id_error = f"ID '{typed_id}' not found in database."
                    else:
                        person   = result
                        id_error = ""
                        typed_id = ""
                        print(f"[LOGIN] Found: {person['Name']} ({person['ID']})")
                        _clear_face_data()
                        state        = S_CAPTURE
                        cap_phase    = 0
                        cap_cd_start = now
                        cap_cd_secs  = 3
                        show_no_face = False

            elif key == 8:                             # BACKSPACE
                typed_id = typed_id[:-1]
                id_error = ""

            elif 32 <= key <= 126:                     # printable ASCII
                if len(typed_id) < 12:
                    typed_id += chr(key).upper()
                    id_error  = ""

        # ══════════════════════════════════════════════════
        #  CAPTURE SCREEN
        # ══════════════════════════════════════════════════
        elif state == S_CAPTURE:
            elapsed   = now - cap_cd_start
            countdown = max(0, cap_cd_secs - int(elapsed))

            render_capture(frame, person, cap_phase, countdown)

            if show_no_face:
                draw_text(frame, "No face detected - move closer and retry",
                          (fw // 2 - 230, fh // 2 - 210),
                          scale=0.65, color=CLR_RED)

            if key == 27:                              # ESC → back to login
                _clear_face_data()
                state    = S_LOGIN
                typed_id = ""
                id_error = ""

            elif elapsed >= cap_cd_secs:
                # Time to snap!
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)

                if len(encs) == 0:
                    # No face — restart countdown for this phase
                    show_no_face = True
                    cap_cd_start = now
                    cap_cd_secs  = 2
                else:
                    show_no_face = False
                    _ram_encodings.append(encs[0])
                    _ram_photos.append(frame.copy())
                    print(f"[CAPTURE] Photo {cap_phase + 1}/3 saved to RAM.")
                    cap_phase   += 1
                    cap_cd_start = now
                    cap_cd_secs  = 3

                    if cap_phase >= 3:
                        # All 3 photos done → start tracking
                        print(f"[TRACKING] Starting for {person['Name']} ...")
                        log_event(person["ID"], person["Name"], "SESSION_START")
                        state           = S_TRACKING
                        is_present      = False
                        body_only       = False
                        missing_since   = None
                        session_start   = now
                        alert_count     = 0
                        alert_flash     = False
                        flash_until     = 0.0
                        last_alert_time = 0.0
                        face_locs       = []
                        frame_idx       = 0
                        body_hits       = 0

        # ══════════════════════════════════════════════════
        #  TRACKING SCREEN
        # ══════════════════════════════════════════════════
        elif state == S_TRACKING:

            if key in (ord('e'), ord('E')):
                state = S_ENDING

            elif key == 27:                            # ESC → back to login
                elapsed = int(now - session_start)
                log_event(person["ID"], person["Name"], "SESSION_END",
                          f"ESC after {elapsed}s  alerts={alert_count}")
                _clear_face_data()
                state    = S_LOGIN
                typed_id = ""
                id_error = ""
                person   = None

            else:
                # ── Face recognition (every FR_EVERY frames) ─────
                if frame_idx % FR_EVERY == 0:
                    small     = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    raw_locs  = face_recognition.face_locations(rgb_small)

                    # Scale locations back to full frame
                    face_locs = [(t*2, r*2, b*2, l*2)
                                 for t, r, b, l in raw_locs]

                    found = False
                    if raw_locs and _ram_encodings:
                        encs = face_recognition.face_encodings(rgb_small, raw_locs)
                        for enc in encs:
                            dists = face_recognition.face_distance(
                                _ram_encodings, enc)
                            if len(dists) > 0 and float(np.min(dists)) <= config.FACE_TOLERANCE:
                                found = True
                                break

                    if found:
                        is_present    = True
                        body_only     = False
                        missing_since = None
                        body_hits     = 0
                    else:
                        is_present = False

                # ── MediaPipe body fallback — every MP_EVERY frames ───
                if not is_present and frame_idx % MP_EVERY == 0:
                    mp_detected = person_in_frame_mediapipe(frame)
                    if mp_detected:
                        body_hits = min(body_hits + 1, BODY_CONFIRM)
                    else:
                        body_hits = max(body_hits - 1, 0)

                if not is_present:
                    # Only confirm "back turned" after BODY_CONFIRM consecutive hits
                    body_only = (body_hits >= BODY_CONFIRM)

                    if body_only:
                        # Body confirmed — person turned around, NOT missing
                        missing_since = None
                    else:
                        # No face, no confirmed body → start missing timer
                        if missing_since is None:
                            missing_since = now

                # ── Alert logic ───────────────────────────────────
                if (missing_since is not None
                        and not is_present
                        and not body_only):
                    miss_dur = now - missing_since
                    cooldown = getattr(config, "ALERT_COOLDOWN_SEC", 60)
                    if (miss_dur >= config.MISSING_THRESHOLD_SEC
                            and now - last_alert_time >= cooldown):
                        snapshot        = frame.copy()
                        alert_count    += 1
                        last_alert_time = now
                        alert_flash     = True
                        flash_until     = now + 5.0
                        print(f"[ALERT #{alert_count}] {person['Name']} "
                              f"missing {int(miss_dur)}s → emailing ...")
                        log_event(person["ID"], person["Name"],
                                  "MISSING_ALERT", f"missing {int(miss_dur)}s")
                        send_alert_email(person, missing_since, snapshot)

            # Render tracking HUD
            render_tracking(frame, person, is_present, body_only,
                            missing_since, session_start, face_locs,
                            alert_count, alert_flash, flash_until)

            # Clear flash flag after timer expires
            if alert_flash and now >= flash_until:
                alert_flash = False

        # ══════════════════════════════════════════════════
        #  END SESSION SCREEN
        # ══════════════════════════════════════════════════
        elif state == S_ENDING:
            # Render tracking underneath for context
            render_tracking(frame, person, is_present, body_only,
                            missing_since, session_start, face_locs,
                            alert_count, False, 0)
            render_ending(frame, person, session_start, alert_count)

            if key in (13, 10):                        # ENTER → confirm
                elapsed = int(now - session_start)
                log_event(person["ID"], person["Name"], "SESSION_END",
                          f"duration {elapsed}s  alerts={alert_count}")
                print(f"[SESSION ENDED] {person['Name']} | "
                      f"{elapsed}s | alerts={alert_count}")
                _clear_face_data()
                state    = S_LOGIN
                typed_id = ""
                id_error = ""
                person   = None

            elif key == 27:                            # ESC → back to tracking
                state = S_TRACKING

        # ── Display frame ──────────────────────────────────
        cv2.imshow("Smart Presence Tracker", frame)

    # ══════════════════════════════════════════════════════
    #  CLEANUP ON EXIT
    # ══════════════════════════════════════════════════════
    _clear_face_data()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[EXIT] Camera released. Face data cleared. Goodbye.\n")


if __name__ == "__main__":
    main()