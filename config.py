# ══════════════════════════════════════════════════
#  SMART PRESENCE TRACKER — CONFIGURATION
# ══════════════════════════════════════════════════

# ── Gmail Credentials ──────────────────────────────
SENDER_EMAIL    = "jamestempmail007@gmail.com"      # Your Gmail
SENDER_PASSWORD = "ljko kbai kjvp yfpc"       # Gmail App Password (16 chars)

# ── How to get Gmail App Password ─────────────────
#  1. Go to  myaccount.google.com
#  2. Security → 2-Step Verification → Enable it
#  3. Security → App Passwords → Create one
#  4. Paste the 16-character code above
# ──────────────────────────────────────────────────

# ── Tracking Settings ─────────────────────────────
MISSING_THRESHOLD_SEC = 30    # seconds before email alert fires
FACE_TOLERANCE        = 0.50  # 0.4 = strict  |  0.6 = lenient
ALERT_COOLDOWN_SEC    = 60    # min seconds between repeated alerts