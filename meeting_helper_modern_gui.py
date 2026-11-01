import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None
from PySide6.QtCore import (
    Qt, QTimer, QPointF, QRectF, QPropertyAnimation, QEasingCurve, Signal, QObject,
    QRect, QPoint,
)
from PySide6.QtGui import (
    QColor, QPainter, QPainterPath, QLinearGradient, QRadialGradient,
    QFont, QFontMetrics, QPen, QBrush, QCursor, QClipboard, QTextCursor,
    QIcon, QPixmap, QFontDatabase,
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QSizePolicy, QGraphicsDropShadowEffect, QSizeGrip,
    QDialog, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QScrollArea,
    QFrame, QListWidget, QListWidgetItem, QAbstractItemView,
    QStackedWidget,
)

from config_manager import (
    APP_VERSION,
    CONFIG_PATH,
    DEFAULT_SETTINGS,
    delete_local_history,
    log_path,
    normalize_settings,
    save_config,
)
from meeting_helper import (
    audio_to_wav_bytes,
    build_llm_request,
    ConversationHistory,
    get_system_audio_device,
    load_settings,
    mix_audio_tracks,
    parse_llm_response,
    record_system_audio_loopback,
    resample_audio,
    save_history_entry,
)

SAMPLE_RATE = 16000
CHANNELS = 1
APP_NAME = "PySufler"
APP_DISPLAY_NAME = "PySufler — Суфлёр совещаний"
APP_ID = "PySufler"


def system_font_family():
    families = QFontDatabase.families()
    return next((name for name in ("Segoe UI Variable", "Segoe UI") if name in families), QApplication.font().family())
PROJECT_DIR = Path(__file__).resolve().parent
ICON_PATH = PROJECT_DIR / "meeting_helper.ico"

# Palette
BG        = QColor(5, 10, 22)
SHELL     = QColor(5, 10, 22)
GLASS     = QColor(11, 18, 32)
GLASS_ALT = QColor(14, 24, 48)
TEXT      = QColor(232, 237, 245)
MUTED     = QColor(122, 144, 176)
CYAN      = QColor(19, 200, 255)
CYAN_DIM  = QColor(10, 74, 110)
PURPLE    = QColor(141, 77, 255)
PURPLE_DIM = QColor(46, 26, 102)
PINK      = QColor(255, 79, 216)
GREEN     = QColor(34, 197, 94)
RED       = QColor(251, 79, 122)
RED_DIM   = QColor(90, 20, 40)

STATUS_BG = QColor(8, 15, 30)
DIAGNOSTIC_LOG = log_path("diagnostics.log")


def env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on", "да")


def lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    return QColor(
        int(c1.red()   + (c2.red()   - c1.red())   * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue()  + (c2.blue()  - c1.blue())  * t),
    )


# ── Signals bridge (thread→UI) ────────────────────────────────────────────────

class _Bridge(QObject):
    set_status_sig      = Signal(str)
    set_question_sig    = Signal(str)
    set_answer_sig      = Signal(str)
    append_question_sig = Signal(str)

_bridge = _Bridge()


def _normalize_transcription_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clean_live_transcription_text(text: str) -> str:
    text = _normalize_transcription_text(text)
    # Whisper sometimes emits "..." as a no-speech or unfinished-preview marker.
    if re.fullmatch(r"[.\s…]+", text):
        return ""
    return re.sub(r"\s*\.{3,}\s*$", "", text).strip()


def _word_key(word: str) -> str:
    return re.sub(r"^[^\wа-яА-ЯёЁ]+|[^\wа-яА-ЯёЁ]+$", "", word, flags=re.UNICODE).lower()


def _has_meaningful_text(text: str) -> bool:
    return bool(re.search(r"[\wа-яА-ЯёЁ]", text or "", flags=re.UNICODE))


def _dedupe_boundary_words(existing: str, tail: str, max_words=4) -> str:
    existing_words = existing.split()
    tail_words = tail.split()
    if not existing_words or not tail_words:
        return tail
    existing_keys = [_word_key(word) for word in existing_words]
    tail_keys = [_word_key(word) for word in tail_words]
    for size in range(min(max_words, len(existing_words), len(tail_words)), 0, -1):
        if existing_keys[-size:] == tail_keys[:size]:
            return " ".join(tail_words[size:])
    return tail


def merge_transcription_chunks(existing: str, new_chunk: str) -> str:
    """Merge Whisper chunks without truncating text or duplicating overlap."""
    existing = _normalize_transcription_text(existing)
    new_chunk = _clean_live_transcription_text(new_chunk)
    if not _has_meaningful_text(new_chunk):
        return existing
    if not existing:
        return new_chunk
    if new_chunk == existing or new_chunk in existing:
        return existing
    if new_chunk.startswith(existing):
        return new_chunk
    if existing.endswith(new_chunk):
        return existing

    existing_words = existing.split()
    new_words = new_chunk.split()
    existing_keys = [_word_key(word) for word in existing_words]
    new_keys = [_word_key(word) for word in new_words]

    max_overlap = min(len(existing_words), len(new_words))
    for size in range(max_overlap, 0, -1):
        if existing_keys[-size:] == new_keys[:size]:
            tail = " ".join(new_words[size:])
            return f"{existing} {tail}".strip() if tail else existing

    tail = _dedupe_boundary_words(existing, new_chunk)
    return f"{existing} {tail}".strip() if tail else existing


def log_whisper_diagnostic(enabled: bool, event: str, **metrics):
    """Write technical metadata only; conversation content is never logged."""
    if not enabled:
        return
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        allowed = {
            "raw_bytes", "extracted_chars", "cleaned_chars", "merged_chars",
            "status_code", "error_type",
        }
        safe_metrics = " ".join(
            f"{key}={value}"
            for key, value in metrics.items()
            if key in allowed
            and (isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 40))
        )
        with open(DIAGNOSTIC_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] event={event} {safe_metrics}\n".rstrip() + "\n")
    except OSError:
        pass


# ── Animated background ────────────────────────────────────────────────────────

class NeonBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.phase = 0.0
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(70)

    def _tick(self):
        self.phase = (self.phase + 0.014) % (math.pi * 2)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, w, h), 28, 28)
        p.setClipPath(clip)

        # Base gradient
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(5, 8, 20))
        grad.setColorAt(1.0, QColor(10, 15, 38))
        p.fillRect(0, 0, w, h, grad)

        pulse = 0.8 + 0.2 * math.sin(self.phase)

        # Glow orbs
        def _orb(cx, cy, rx, ry, color: QColor, alpha_mul: float):
            rg = QRadialGradient(cx, cy, max(rx, ry))
            c1 = QColor(color)
            c1.setAlpha(int(90 * alpha_mul * pulse))
            c2 = QColor(color)
            c2.setAlpha(0)
            rg.setColorAt(0.0, c1)
            rg.setColorAt(1.0, c2)
            p.save()
            p.translate(cx, cy)
            p.scale(rx / max(ry, 1), 1.0)
            p.fillRect(QRectF(-ry, -ry, ry * 2, ry * 2), rg)
            p.restore()

        _orb(w * 0.10, h * 0.18, int(w * 0.48), int(h * 0.48), QColor(10, 164, 255), 0.55)
        _orb(w * 0.80, h * 0.22, int(w * 0.45), int(h * 0.50), QColor(110, 40, 255), 0.40)
        _orb(w * 0.75, h * 0.82, int(w * 0.38), int(h * 0.38), QColor(204, 60, 166), 0.32)
        _orb(w * 0.12, h * 0.75, int(w * 0.32), int(h * 0.36), QColor(8, 63, 123), 0.36)

        # Diagonal glass streaks
        for off, alpha in ((-w * 0.18, 18), (w * 0.22, 12), (w * 0.62, 8)):
            path = QPainterPath()
            path.moveTo(off, 0)
            path.lineTo(off + w * 0.26, 0)
            path.lineTo(off + w * 0.06, h)
            path.lineTo(off - w * 0.20, h)
            path.closeSubpath()
            c = QColor(255, 255, 255, alpha)
            p.fillPath(path, c)

        p.end()


# ── RingIcon ───────────────────────────────────────────────────────────────────

class RingIcon(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(54, 54)
        self._angle = 0.0   # degrees, grows continuously 0..360 then wraps
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(16)

    def _tick(self):
        self._angle = (self._angle + 1.5) % 360.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(0, 0, 54, 54, Qt.transparent)
        cx = cy = 27

        # Pulsing glow ring
        pulse = 3 + math.sin(math.radians(self._angle)) * 1.8
        p.setPen(QPen(QColor(20, 40, 90, 80), 6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), 22 + pulse, 22 + pulse)

        rect = QRectF(7, 7, 40, 40)

        # Spinning arc — purple
        pen = QPen(PURPLE, 3.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int(self._angle * 16), 220 * 16)

        # Spinning arc — cyan (offset 175°)
        pen = QPen(CYAN, 3.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, int((self._angle + 175) * 16), 130 * 16)

        # Centre dot
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(14, 232, 255))
        p.drawEllipse(QPointF(cx, cy), 4, 4)
        p.end()


# ── NeonButton ─────────────────────────────────────────────────────────────────

class NeonButton(QWidget):
    clicked = Signal()

    def __init__(self, text, accent: QColor, dim: QColor, parent=None, icon_kind=None):
        super().__init__(parent)
        self.setFixedSize(188, 54)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._text   = text
        self._icon_kind = icon_kind
        self.setAccessibleName(text)
        self.setFocusPolicy(Qt.StrongFocus)
        self._accent = accent
        self._dim    = dim
        self._glow   = 0.0      # 0..1
        self._hover  = False

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick_glow)
        self._anim.start(16)

    # Public interface matches old GlowButton.configure()
    def configure(self, **kw):
        if "text" in kw:
            self._text = kw["text"]
            self.setAccessibleName(self._text)
        if "fg_color" in kw:
            c = kw["fg_color"]
            if isinstance(c, str):
                self._dim = QColor(c)
            else:
                self._dim = c
        if "border_color" in kw:
            c = kw["border_color"]
            if isinstance(c, str):
                self._accent = QColor(c)
            else:
                self._accent = c
        self.update()

    def set_size(self, w, h):
        self.setFixedSize(w, h)

    def _tick_glow(self):
        target = 1.0 if self._hover else 0.0
        self._glow += (target - self._glow) * 0.14
        self.update()

    def enterEvent(self, _):
        self._hover = True

    def leaveEvent(self, _):
        self._hover = False

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if not event.isAutoRepeat():
                self.clicked.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)
        w, h, r = self.width(), self.height(), 16

        # Outer glow halo
        for i in range(8, 0, -1):
            pad = float((8 - i) * 2.2)
            alpha = int(self._glow * 60 * (i / 8))
            glow_c = QColor(self._accent)
            glow_c.setAlpha(alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(glow_c)
            rect = QRectF(pad, pad, w - pad * 2, h - pad * 2)
            p.drawRoundedRect(rect, r + pad * 0.3, r + pad * 0.3)

        # Body fill
        body_col = lerp_color(self._dim, self._accent.darker(280), self._glow)
        p.setBrush(body_col)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(3, 3, w - 6, h - 6), r, r)

        # Top highlight shimmer
        hl = QLinearGradient(0, 3, 0, h // 2)
        hl.setColorAt(0.0, QColor(255, 255, 255, int(30 + self._glow * 20)))
        hl.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(hl)
        p.drawRoundedRect(QRectF(3, 3, w - 6, (h - 6) // 2), r, r)

        # Border
        pen = QPen(self._accent, 1.2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(2, 2, w - 4, h - 4), r, r)
        if self.hasFocus():
            p.setPen(QPen(TEXT, 1, Qt.DotLine))
            p.drawRoundedRect(QRectF(6, 6, w - 12, h - 12), r - 4, r - 4)

        # Text
        p.setPen(TEXT)
        font = QFont(system_font_family(), 12, QFont.Bold)
        p.setFont(font)
        text_rect = QRectF(0, 0, w, h)
        if self._icon_kind:
            text_width = QFontMetrics(font).horizontalAdvance(self._text)
            left = (w - text_width - 26) / 2
            if not self._text:
                left = (w - 14) / 2
            p.setPen(QPen(TEXT, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if self._icon_kind == "record":
                if self._text.strip() == "Стоп":
                    p.fillRect(QRectF(left, h / 2 - 6, 12, 12), RED)
                else:
                    p.setBrush(CYAN)
                    p.drawEllipse(QPointF(left + 6, h / 2), 5, 5)
            elif self._icon_kind == "check":
                p.drawLine(QPointF(left, h / 2), QPointF(left + 5, h / 2 + 5))
                p.drawLine(QPointF(left + 5, h / 2 + 5), QPointF(left + 14, h / 2 - 5))
            elif self._icon_kind == "answer":
                path = QPainterPath(QPointF(left + 9, h / 2 - 9))
                for x, y in ((0, 2), (6, 2), (4, 9), (15, -3), (8, -3)):
                    path.lineTo(left + x, h / 2 + y)
                path.closeSubpath()
                p.drawPath(path)
            elif self._icon_kind == "clear":
                p.drawRoundedRect(QRectF(left + 2, h / 2 - 4, 10, 12), 2, 2)
                p.drawLine(QPointF(left, h / 2 - 6), QPointF(left + 14, h / 2 - 6))
                p.drawLine(QPointF(left + 5, h / 2 - 9), QPointF(left + 9, h / 2 - 9))
            elif self._icon_kind == "settings":
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(left + 7, h / 2), 5, 5)
                p.drawEllipse(QPointF(left + 7, h / 2), 1.5, 1.5)
                for i in range(8):
                    a = i * math.pi / 4
                    p.drawLine(QPointF(left + 7 + 6 * math.cos(a), h / 2 + 6 * math.sin(a)),
                               QPointF(left + 7 + 8 * math.cos(a), h / 2 + 8 * math.sin(a)))
            text_rect = QRectF(left + 26, 0, text_width + 2, h)
        p.drawText(text_rect, Qt.AlignCenter, self._text)
        p.end()


# ── GlassCard ─────────────────────────────────────────────────────────────────

class GlassCard(QWidget):
    def __init__(self, accent: QColor, parent=None):
        super().__init__(parent)
        self._accent = accent
        self.setContentsMargins(12, 12, 12, 12)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, r = self.width(), self.height(), 26

        # Glow halo
        for i in range(7, 0, -1):
            pad = float((7 - i) * 1.7)
            alpha = int(13 * (i / 7))
            gc = QColor(self._accent)
            gc.setAlpha(alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(gc)
            p.drawRoundedRect(QRectF(pad, pad, w - pad * 2, h - pad * 2),
                              r + pad * 0.3, r + pad * 0.3)

        # Glass body
        body = QLinearGradient(0, 0, 0, h)
        body.setColorAt(0.0, QColor(18, 30, 58, 206))
        body.setColorAt(1.0, QColor(13, 22, 44, 206))
        p.setBrush(body)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(4, 4, w - 8, h - 8), r, r)

        # Border
        border = QColor(self._accent)
        border.setAlpha(115)
        border_pen = QPen(border, 1.0)
        p.setPen(border_pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(4, 4, w - 8, h - 8), r, r)
        p.end()


# ── NeonWaveform ───────────────────────────────────────────────────────────────

class NeonWaveform(QWidget):
    def __init__(self, is_recording_fn, level_fn, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._is_rec = is_recording_fn
        self._level  = level_fn
        self.phase   = 0.0
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(32)

    def _tick(self):
        self.phase += 0.21 if self._is_rec() else 0.06
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        active = self._is_rec()
        level = max(0.0, min(1.0, self._level()))

        # Soft compact glass strip instead of a heavy black panel.
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(6, 12, 26, 70 if active else 35))
        p.drawRoundedRect(QRectF(22, 8, w - 44, h - 16), 18, 18)

        # Faint grid lines
        for gy in (mid - 22, mid, mid + 22):
            pen = QPen(QColor(22, 44, 72, 85 if active else 45), 1)
            p.setPen(pen)
            p.drawLine(QPointF(50, gy), QPointF(w - 50, gy))

        # Bars
        p.setRenderHint(QPainter.Antialiasing, False)
        bars = []
        for i, x in enumerate(range(54, w - 54, 6)):
            env = math.exp(-((x - w * 0.5) ** 2) / (2 * (w * 0.18) ** 2))
            voice = level if active else 0.015
            ripple = 0.28 + 0.72 * abs(math.sin(i * 0.32 + self.phase))
            breath = 0.72 + 0.28 * math.sin(self.phase * 0.6 + i * 0.05)
            amp = (6 + voice * 56) * env * ripple * breath + (4 if active else 1.5)
            t = (x - 54) / max(1, w - 108)
            if t < 0.35:
                col = CYAN
            elif t < 0.65:
                col = PURPLE
            else:
                col = PINK
            bars.append((x, amp, col))

        # Deep shadow pass
        p.setPen(QPen(QColor(13, 22, 56, 170 if active else 80), 9))
        for x, amp, _ in bars:
            p.drawLine(QPointF(x, mid - amp - 3), QPointF(x, mid + amp + 3))

        # Dim glow pass
        for x, amp, col in bars:
            dc = QColor(col.red() // 4, col.green() // 4, col.blue() // 4,
                        210 if active else 120)
            p.setPen(QPen(dc, 5))
            p.drawLine(QPointF(x, mid - amp - 1), QPointF(x, mid + amp + 1))

        # Bright neon pass
        for x, amp, col in bars:
            bright = QColor(col)
            bright.setAlpha(235 if active else 130)
            p.setPen(QPen(bright, 2))
            p.drawLine(QPointF(x, mid - amp), QPointF(x, mid + amp))

        # Baseline
        p.setPen(QPen(QColor(26, 48, 96, 180), 1))
        p.drawLine(QPointF(50, mid), QPointF(w - 50, mid))
        p.end()


# ── ToggleSwitch ───────────────────────────────────────────────────────────────

class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked=True, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 24)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._checked = checked
        self._margin = 3.0
        self._thumb = 18.0
        self._knob_x = self._target_x()

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._anim.start(16)

    def isChecked(self): return self._checked

    def setChecked(self, v):
        if v != self._checked:
            self._checked = v
            self.toggled.emit(v)

    def _target_x(self):
        return self.width() - self._margin - self._thumb if self._checked else self._margin

    def _tick(self):
        target = self._target_x()
        self._knob_x += (target - self._knob_x) * 0.22
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._checked = not self._checked
            self.toggled.emit(self._checked)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        track_col = QColor(22, 171, 219) if self._checked else QColor(38, 48, 66)
        track_c = QColor(track_col)
        track_c.setAlpha(230)
        p.setPen(Qt.NoPen)
        p.setBrush(track_c)
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)

        if self._checked:
            glow = QColor(103, 232, 249, 45)
            p.setBrush(glow)
            p.drawEllipse(QRectF(self._knob_x - 1.5, self._margin - 1.5, self._thumb + 3, self._thumb + 3))
        p.setBrush(QColor(238, 246, 255))
        p.drawEllipse(QRectF(self._knob_x, self._margin, self._thumb, self._thumb))
        p.end()


# ── SoftComboBox ──────────────────────────────────────────────────────────────

class SoftComboBox(QComboBox):
    """Custom dark popup to avoid native white Windows/Qt dropdown frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._popup = None
        self._popup_list = None

    def showPopup(self):
        if self._popup is not None:
            self.hidePopup()

        popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        popup.setAttribute(Qt.WA_TranslucentBackground)
        popup.setObjectName("softComboPopup")

        outer = QVBoxLayout(popup)
        outer.setContentsMargins(7, 7, 7, 7)
        outer.setSpacing(0)

        card = QFrame()
        card.setObjectName("softComboCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)
        card_layout.setSpacing(0)

        list_widget = QListWidget()
        list_widget.setObjectName("softComboList")
        list_widget.setMouseTracking(True)
        list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        list_widget.setFrameShape(QFrame.NoFrame)
        list_widget.setFocusPolicy(Qt.StrongFocus)
        list_widget.setUniformItemSizes(False)

        for index in range(self.count()):
            text = self.itemText(index)
            item = QListWidgetItem(text)
            item.setToolTip(text)
            item.setData(Qt.UserRole, index)
            item.setSizeHint(QRectF(0, 0, 1, 34).size().toSize())
            list_widget.addItem(item)
            if index == self.currentIndex():
                list_widget.setCurrentRow(index)

        list_widget.itemClicked.connect(self._select_popup_item)
        card_layout.addWidget(list_widget)
        outer.addWidget(card)

        popup.setStyleSheet("""
            QFrame#softComboPopup {
                background: transparent;
            }
            QFrame#softComboCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(9, 18, 34, 0.98),
                    stop:0.58 rgba(12, 24, 46, 0.98),
                    stop:1 rgba(26, 18, 58, 0.97));
                border: 1px solid rgba(56, 189, 248, 0.28);
                border-radius: 18px;
            }
            QListWidget#softComboList {
                background: transparent;
                color: #e5e7eb;
                border: none;
                outline: none;
                padding: 2px;
                font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QListWidget#softComboList::item {
                min-height: 32px;
                padding: 7px 10px;
                border-radius: 10px;
                color: #dbeafe;
            }
            QListWidget#softComboList::item:hover {
                background: rgba(56, 189, 248, 0.14);
                color: #ffffff;
            }
            QListWidget#softComboList::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(14, 116, 144, 0.72),
                    stop:1 rgba(109, 40, 217, 0.68));
                color: #ffffff;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 6px 1px 6px 4px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(56, 189, 248, 0.34);
                min-height: 34px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(139, 92, 246, 0.66);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
        """)

        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(56, 189, 248, 58))
        card.setGraphicsEffect(shadow)

        width = self._popup_width()
        height = min(max(72, self.count() * 34 + 30), 340)
        popup.resize(width, height)
        pos = self.mapToGlobal(QPoint(0, self.height() + 6))
        screen = QApplication.screenAt(pos) or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if pos.x() + width > area.right() - 10:
                pos.setX(max(area.left() + 10, area.right() - width - 10))
            if pos.y() + height > area.bottom() - 10:
                pos.setY(max(area.top() + 10, self.mapToGlobal(QPoint(0, -height - 6)).y()))
        popup.move(pos)
        popup.show()
        list_widget.setFocus()
        self._popup = popup
        self._popup_list = list_widget

    def hidePopup(self):
        if self._popup is not None:
            self._popup.close()
            self._popup.deleteLater()
            self._popup = None
            self._popup_list = None

    def _select_popup_item(self, item: QListWidgetItem):
        index = item.data(Qt.UserRole)
        if isinstance(index, int):
            self.setCurrentIndex(index)
        self.hidePopup()

    def _popup_width(self):
        metrics = self.fontMetrics()
        longest = max([metrics.horizontalAdvance(self.itemText(i)) for i in range(self.count())] or [self.minimumWidth()])
        screen_width = 900
        screen = QApplication.screenAt(self.mapToGlobal(QPoint(0, 0))) or QApplication.primaryScreen()
        if screen is not None:
            screen_width = screen.availableGeometry().width()
        return max(self.width(), min(longest + 88, screen_width - 80, 720))

    def refresh_popup_metrics(self):
        for index in range(self.count()):
            text = self.itemText(index)
            self.setItemData(index, text, Qt.ToolTipRole)
        self.setToolTip(self.currentText())

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(138, 206, 238, 210), 1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        x = self.width() - 24
        y = self.height() / 2 - 2
        p.drawLine(QPointF(x, y), QPointF(x + 5, y + 5))
        p.drawLine(QPointF(x + 10, y), QPointF(x + 5, y + 5))
        p.end()


# ── StatusDot ─────────────────────────────────────────────────────────────────

class StatusDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = GREEN

    def set_color(self, c: QColor):
        self._color = c
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(0, 0, 12, 12, Qt.transparent)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(QRectF(1, 1, 10, 10))
        p.end()


# ── GlassStatusBar ────────────────────────────────────────────────────────────

class GlassStatusBar(QWidget):
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(9, 20, 42, 220))
        grad.setColorAt(0.55, QColor(10, 18, 36, 205))
        grad.setColorAt(1.0, QColor(22, 12, 50, 210))
        p.setPen(QPen(QColor(34, 130, 190, 125), 1))
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 20, 20)
        p.end()


# ── CopyButton ────────────────────────────────────────────────────────────────

class CopyButton(QWidget):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 30)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._hover = False

    def enterEvent(self, _): self._hover = True; self.update()
    def leaveEvent(self, _): self._hover = False; self.update()
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bg = QColor(24, 42, 70, 160) if self._hover else QColor(16, 28, 50, 120)
        p.setPen(QPen(QColor(30, 52, 96), 1))
        p.setBrush(bg)
        p.drawRoundedRect(QRectF(0, 0, w, h), 8, 8)
        p.setPen(MUTED)
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "  Копировать")
        p.end()


# ── NeonTextEdit ─────────────────────────────────────────────────────────────

class NeonTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.setCursorWidth(0)
        self.setFrameShape(QTextEdit.NoFrame)
        self.setAcceptRichText(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(14)
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 14)
        self.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #e8edf5;
                placeholder-text-color: rgba(148, 163, 184, 135);
                border: none;
                border-radius: 0px;
                padding: 0px;
                font-family: 'Segoe UI';
                font-size: 18px;
                selection-background-color: #1a4870;
                selection-color: #ffffff;
            }
            QTextEdit:focus { border: none; outline: none; }
            QScrollBar:vertical {
                background: transparent;
                width: 11px;
                margin: 6px 1px 6px 5px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(56, 189, 248, 85);
                min-height: 34px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(139, 92, 246, 170);
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }
        """)
        self.setMinimumHeight(110)
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ── Main Window ───────────────────────────────────────────────────────────────


class SettingsDialog(QDialog):
    FIELD_HEIGHT = 38

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки PySufler")
        self.setWindowIcon(QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon())
        self.setMinimumSize(880, 660)
        self.resize(1020, 800)
        self.settings_result = None
        self._initial_settings = normalize_settings(settings)
        self._initial_api_key = self._initial_settings.get("openai_api_key", "")
        self._build_ui(dict(self._initial_settings))
        available = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(760, 420)
        self.resize(min(1020, available.width() - 32), min(800, available.height() - 32))

    def show_result_dialog(self, title: str, lines, level="info", confirm_text=""):
        status_colors = {"ok": "#5eead4", "warn": "#fbbf24", "error": "#fca5a5", "info": "#93c5fd"}
        dialog = QDialog(self)
        dialog.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        dialog.setAttribute(Qt.WA_TranslucentBackground)
        dialog.setWindowTitle(title)
        dialog.setWindowIcon(QIcon(str(ICON_PATH)) if ICON_PATH.exists() else QIcon())
        dialog.setMinimumSize(540, 300)
        dialog.setStyleSheet("""
            QDialog {
                background: transparent;
                color: #e5e7eb;
                font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QWidget#resultCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #07101f, stop:0.58 #0a1629, stop:1 #11112c);
                border: 1px solid rgba(56, 189, 248, 0.18);
                border-radius: 22px;
            }
            QLabel#resultTitle {
                color: #ffffff;
                font-size: 21px;
                font-weight: 700;
            }
            QLabel#resultText {
                color: #dbeafe;
                background: transparent;
            }
            QLabel#resultDot {
                min-width: 9px;
                max-width: 9px;
                min-height: 9px;
                max-height: 9px;
                border-radius: 4px;
                background: #93c5fd;
            }
            QLabel#resultDot[level="ok"] { background: #5eead4; }
            QLabel#resultDot[level="warn"] { background: #fbbf24; }
            QLabel#resultDot[level="error"] { background: #fca5a5; }
            QPushButton {
                min-height: 38px;
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0e7490, stop:1 #5b21b6);
                border: none;
                border-radius: 13px;
                padding: 0 22px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0891b2, stop:1 #7c3aed);
            }
        """)
        root = QVBoxLayout(dialog)
        root.setContentsMargins(12, 12, 12, 12)
        card = QWidget()
        card.setObjectName("resultCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
        header = QHBoxLayout()
        dot = QLabel()
        dot.setObjectName("resultDot")
        dot.setProperty("level", level)
        header.addWidget(dot)
        title_label = QLabel(title)
        title_label.setObjectName("resultTitle")
        header.addWidget(title_label, 1)
        layout.addLayout(header)
        messages = QWidget()
        messages_layout = QVBoxLayout(messages)
        messages_layout.setContentsMargins(0, 0, 0, 0)
        messages_layout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        messages_scroll = QScrollArea()
        messages_scroll.setWidgetResizable(True)
        messages_scroll.setFrameShape(QFrame.NoFrame)
        messages_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        messages_scroll.setWidget(messages)
        layout.addWidget(messages_scroll, 1)
        for item in lines:
            if isinstance(item, tuple):
                item_level, text = item
            else:
                item_level, text = level, str(item)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            item_dot = QLabel()
            item_dot.setObjectName("resultDot")
            item_dot.setProperty("level", item_level)
            row.addWidget(item_dot, 0, Qt.AlignTop)
            label = QLabel(text)
            label.setObjectName("resultText")
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {status_colors.get(item_level, '#dbeafe')}; background: transparent;")
            row.addWidget(label, 1)
            messages_layout.addLayout(row)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        if confirm_text:
            cancel_btn = QPushButton("Отмена")
            cancel_btn.clicked.connect(dialog.reject)
            buttons.addWidget(cancel_btn)
            confirm_btn = QPushButton(confirm_text)
            confirm_btn.clicked.connect(dialog.accept)
            buttons.addWidget(confirm_btn)
        else:
            close_btn = QPushButton("Закрыть")
            close_btn.clicked.connect(dialog.accept)
            buttons.addWidget(close_btn)
        layout.addLayout(buttons)
        root.addWidget(card)
        available = QApplication.primaryScreen().availableGeometry()
        dialog.resize(min(680, available.width() - 32), min(480, available.height() - 32))
        return dialog.exec() == QDialog.Accepted

    def _build_ui(self, settings: dict):
        self.setStyleSheet(self._settings_stylesheet())
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(900, 620)
        self.resize(960, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)

        shell = QWidget()
        shell.setObjectName("settingsShell")
        root.addWidget(shell)
        self._soft_shadow(shell, blur=42, alpha=90, offset=10)

        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(16, 16, 16, 14)
        shell_layout.setSpacing(12)

        body = QWidget()
        body.setObjectName("settingsBody")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(16)
        shell_layout.addWidget(body, 1)

        sidebar = QWidget()
        sidebar.setObjectName("settingsSidebar")
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(8)
        body_layout.addWidget(sidebar)

        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 10)
        brand_layout.setSpacing(10)
        icon = QLabel()
        icon.setFixedSize(38, 38)
        if ICON_PATH.exists():
            icon.setPixmap(QIcon(str(ICON_PATH)).pixmap(38, 38))
        brand_layout.addWidget(icon)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        brand_title = QLabel("PySufler")
        brand_title.setObjectName("settingsBrandTitle")
        brand_subtitle = QLabel("настройки")
        brand_subtitle.setObjectName("settingsBrandSubtitle")
        brand_text.addWidget(brand_title)
        brand_text.addWidget(brand_subtitle)
        brand_layout.addLayout(brand_text, 1)
        sidebar_layout.addWidget(brand)

        self.nav_buttons = []
        self._page_widgets = []
        self.stack = QStackedWidget()
        self.stack.setObjectName("settingsStack")
        body_layout.addWidget(self.stack, 1)

        self.llm_api_type = self._combo(["openwebui", "openai", "ollama", "ollama_generate"], settings["llm_api_type"], 220)
        self.llm_url = self._line(settings["llm_url"])
        self.llm_model = self._combo([settings["llm_model"]], settings["llm_model"], 230, editable=False)
        self.llm_model.setMaximumWidth(270)
        self.api_key = self._line("", password=True)
        self.api_key.setPlaceholderText(
            "Защищённый ключ сохранён — введите новый для замены"
            if self._initial_api_key else "Введите API-ключ"
        )
        self.api_key.setToolTip("Сохранённое значение не отображается.")
        self.chat_id = self._line(settings["openwebui_chat_id"])
        self.temperature = self._double_spin(0.0, 2.0, float(settings.get("temperature", 0.2)), 0.1)
        self.max_tokens = self._spin(1, 8192, int(settings.get("max_tokens", 300)))
        self.llm_timeout = self._spin(5, 1200, int(settings["llm_timeout"]))

        refresh_btn = self._button("Обновить список", "secondary")
        refresh_btn.clicked.connect(self.refresh_models)
        self.model_status = QLabel("")
        self.model_status.setObjectName("settingsInlineStatus")
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(10)
        model_row.addWidget(self.llm_model, 1)
        model_row.addWidget(refresh_btn)

        if not settings["system_audio_enabled"]:
            audio_summary = "микрофон"
        elif settings["mix_mic_and_system_audio"]:
            audio_summary = "микрофон + системный звук"
        else:
            audio_summary = "системный звук"
        overview = self._section_page("Основные", "Краткая сводка локального помощника и быстрые действия.")
        overview.addWidget(self._info_card("PySufler — Суфлёр совещаний", f"Версия {APP_VERSION}. Локальное приложение для записи микрофона и системного звука, live-распознавания и коротких ответов через локальную LLM."))
        overview.addWidget(self._status_card("Текущая модель", settings["llm_model"], "LLM"))
        overview.addWidget(self._status_card("Режим звука", audio_summary, "Audio"))
        instruction_btn = self._button("Открыть инструкцию", "secondary")
        instruction_btn.clicked.connect(lambda: self._switch_section(7))
        overview.addWidget(instruction_btn)
        overview.addStretch(1)
        self._add_page("general", "Основные", overview)

        llm_page = self._section_page("Модель и LLM", "Endpoint, модель и параметры генерации ответа.")
        llm_page.addWidget(self._group("Подключение", [
            self._setting_row("Тип API", "Формат локального LLM endpoint.", self.llm_api_type),
            self._setting_row("Адрес LLM", "OpenWebUI, OpenAI-compatible или Ollama endpoint.", self.llm_url),
            self._setting_row("Модель", "Текущая целевая модель сохраняется при обновлении списка.", model_row),
            self._setting_row("API-ключ", "Хранится через Windows DPAPI и никогда не отображается повторно.", self.api_key),
            self._setting_row("OpenWebUI chat_id", "Необязательно. Оставьте пустым для обычных запросов; иначе нужен ID существующего чата.", self.chat_id),
        ]))
        generation_row = QHBoxLayout()
        generation_row.setContentsMargins(0, 0, 0, 0)
        generation_row.setSpacing(12)
        generation_row.addWidget(self._mini_field("Temperature", self.temperature))
        generation_row.addWidget(self._mini_field("Max tokens", self.max_tokens))
        generation_row.addWidget(self._mini_field("Timeout, сек", self.llm_timeout))
        generation_row.addStretch(1)
        llm_page.addWidget(self._group("Генерация", [
            self._setting_row("Параметры", "Температура, лимит ответа и timeout запроса.", generation_row),
        ]))
        llm_page.addWidget(self._info_card("Передача данных", "В LLM отправляются текущий текст и до пяти предыдущих пар вопрос–ответ. Для адресов вне этого компьютера используйте HTTPS."))
        llm_page.addWidget(self.model_status)
        llm_page.addStretch(1)
        self._add_page("llm", "Модель и LLM", llm_page)

        self.whisper_url = self._line(settings["whisper_url"])
        self.language = self._line(settings["language"])
        self.chunk_seconds = self._spin(1, 60, int(settings["chunk_seconds"]))
        self.buffer_minutes = self._spin(1, 240, int(settings["buffer_minutes"]))
        whisper_page = self._section_page("Whisper", "Настройки локального STT endpoint и размеров фрагментов.")
        whisper_page.addWidget(self._group("Распознавание", [
            self._setting_row("Адрес Whisper", "OpenAI-compatible endpoint транскрипции.", self.whisper_url),
            self._setting_row("Язык", "Код языка распознавания, обычно ru.", self.language),
            self._setting_row("Длина фрагмента", "Размер аудио-куска для фонового режима, сек.", self.chunk_seconds),
            self._setting_row("Буфер текста", "Сколько минут текста держать в памяти.", self.buffer_minutes),
        ]))
        whisper_page.addWidget(self._info_card("Передача аудио", "Whisper получает WAV с выбранных источников. Для удалённого или сетевого endpoint используйте HTTPS."))
        whisper_page.addStretch(1)
        self._add_page("whisper", "Whisper", whisper_page)

        self.input_device = self._combo([], None, 520)
        self.input_device.addItem("Авто", None)
        self.system_audio_device = self._combo([], None, 520)
        self.system_audio_device.addItem("Авто WASAPI loopback", None)
        self._fill_audio_devices(settings)
        self.system_audio_enabled, system_audio_enabled_row = self._toggle_row(
            "Включить системный звук", bool(settings["system_audio_enabled"])
        )
        self.auto_system_loopback, auto_system_loopback_row = self._toggle_row(
            "Автоматический выбор WASAPI loopback", bool(settings.get("auto_system_loopback", True))
        )
        self.mix_audio, mix_audio_row = self._toggle_row(
            "Объединять микрофон и системный звук", bool(settings["mix_mic_and_system_audio"])
        )
        audio_page = self._section_page("Аудио", "Микрофон, системный звук и автоматический Windows WASAPI loopback.")
        self._set_control_width(self.input_device, 300)
        self._set_control_width(self.system_audio_device, 300)
        audio_page.addWidget(self._group("Источники", [
            self._setting_row("Микрофон", "Устройство записи пользователя. Авто использует системное значение.", self.input_device),
            self._setting_row("Системный звук", "Включает запись голоса собеседника из Teams, Zoom, Meet или браузера.", self._switch_control(self.system_audio_enabled)),
            self._setting_row("Автовыбор WASAPI", "Автоматически подобрать loopback-устройство Windows.", self._switch_control(self.auto_system_loopback)),
            self._setting_row("Устройство loopback", "Можно оставить Авто, если определение работает.", self.system_audio_device),
            self._setting_row("Объединять источники", "Перед Whisper отправляется общий WAV: микрофон + системный звук.", self._switch_control(self.mix_audio)),
        ]))
        audio_page.addStretch(1)
        self._add_page("audio", "Аудио", audio_page)

        self.live_transcription, live_transcription_row = self._toggle_row(
            "Live-распознавание", bool(settings["live_transcription"])
        )
        self.live_chunk_seconds = self._spin(1, 20, int(settings["live_chunk_seconds"]))
        live_page = self._section_page("Live-режим", "Промежуточный текст во время записи. Финальный текст формируется после остановки.")
        live_page.addWidget(self._group("Промежуточное распознавание", [
            self._setting_row("Live-распознавание", "Показывать промежуточный текст в поле вопроса.", live_transcription_row),
            self._setting_row("Live-фрагмент", "Размер аудио-куска для live-preview, сек.", self.live_chunk_seconds),
        ]))
        live_page.addWidget(self._info_card("Важно", "В LLM текст отправляется только после нажатия кнопки «Ответ». Live-режим не отправляет каждый фрагмент в модель."))
        live_page.addStretch(1)
        self._add_page("live", "Live-режим", live_page)

        self.hotkey = self._line(settings["hotkey"])
        self.prompt_mode = self._combo(["last", "buffer"], settings["prompt_mode"], 220)
        self.last_lines = self._spin(1, 100, int(settings["last_lines"]))
        self.save_history = ToggleSwitch(bool(settings.get("save_history", False)))
        self.diagnostic_mode = ToggleSwitch(bool(settings.get("diagnostic_mode", False)))
        advanced_page = self._section_page("Дополнительно", "Горячая клавиша, режим промпта и локальные каталоги.")
        open_config_btn = self._button("Открыть папку конфигурации", "secondary")
        open_config_btn.clicked.connect(self._open_config_folder)
        delete_history_btn = self._button("Удалить локальную историю", "secondary")
        delete_history_btn.clicked.connect(self._delete_local_history)
        advanced_page.addWidget(self._group("Поведение", [
            self._setting_row("Горячая клавиша", "Команда для консольного режима meeting_helper.py.", self.hotkey),
            self._setting_row("Режим промпта", "last отправляет последние фразы, buffer — весь буфер.", self.prompt_mode),
            self._setting_row("Последних фраз", "Количество непустых фраз для режима last.", self.last_lines),
        ]))
        advanced_page.addWidget(self._group("Данные", [
            self._setting_row("Сохранять историю", "Записывать transcript.txt и suggestions.txt. По умолчанию выключено.", self._switch_control(self.save_history)),
            self._setting_row("Диагностика", "Записывать только технические метрики без текста разговоров.", self._switch_control(self.diagnostic_mode)),
            self._setting_row("Конфигурация", self._elide_path(str(CONFIG_PATH)), open_config_btn),
            self._setting_row("Локальная история", "Удаляет сохранённые расшифровки, ответы и старый приватный debug-log.", delete_history_btn),
        ]))
        advanced_page.addStretch(1)
        self._add_page("advanced", "Дополнительно", advanced_page)

        diagnostics_page = self._section_page("Диагностика", "Проверка доступности LLM, моделей и Whisper endpoint.")
        diag_btn = self._button("Проверить подключение", "primary")
        diag_btn.clicked.connect(self.check_connection)
        diag_actions = QHBoxLayout()
        diag_actions.setContentsMargins(0, 0, 0, 0)
        diag_actions.setSpacing(10)
        diag_actions.addWidget(diag_btn)
        diag_actions.addStretch(1)
        diagnostics_page.addWidget(self._group("Проверки", [
            self._setting_row("Сервисы", "LLM endpoint, список моделей и Whisper endpoint.", diag_actions),
            self._setting_row("Целевая модель", "Используется qwen3.5:9b. Старые значения по умолчанию не применяются.", self._pill(settings["llm_model"])),
        ]))
        diagnostics_page.addStretch(1)
        self._add_page("diagnostics", "Диагностика", diagnostics_page)

        instruction_page = self._section_page("Инструкция", "Короткая встроенная памятка по работе с PySufler.")
        instruction_page.addWidget(self._info_card("Как пользоваться", "1. Включите системный звук, если нужно слышать собеседника.\n2. Нажмите «Запись» в главном окне.\n3. Остановите запись.\n4. Проверьте распознанный текст.\n5. Нажмите «Ответ».\n6. Используйте подсказку помощника."))
        instruction_page.addWidget(self._info_card("Live-режим", "Промежуточный текст появляется во время записи. После остановки полное распознавание заменяет live-preview финальным текстом."))
        instruction_page.addStretch(1)
        self._add_page("help", "Инструкция", instruction_page)

        bottom = QWidget()
        bottom.setObjectName("settingsFooter")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(16, 12, 16, 12)
        bottom_layout.setSpacing(10)
        defaults_btn = self._button("По умолчанию", "secondary")
        defaults_btn.clicked.connect(self.restore_defaults)
        save_btn = self._button("Сохранить", "primary")
        save_btn.clicked.connect(self.accept)
        cancel_btn = self._button("Отмена", "secondary")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(defaults_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(save_btn)
        bottom_layout.addWidget(cancel_btn)
        shell_layout.addWidget(bottom)

        sidebar_layout.addStretch(1)
        self._switch_section(0)
        self._refresh_combo_metrics()

    def _soft_shadow(self, widget: QWidget, blur=28, alpha=55, offset=6):
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, offset)
        shadow.setColor(QColor(0, 0, 0, alpha))
        widget.setGraphicsEffect(shadow)

    def _add_page(self, icon: str, title: str, content_layout: QVBoxLayout):
        page = content_layout.parentWidget()
        scroll = QScrollArea()
        scroll.setObjectName("settingsPageScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content_layout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        scroll.setWidget(page)
        index = self.stack.addWidget(scroll)

        button = QPushButton(title)
        button.setObjectName("settingsNavItem")
        button.setIcon(self._nav_icon(icon))
        button.setIconSize(QRectF(0, 0, 16, 16).size().toSize())
        button.setCheckable(True)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.clicked.connect(lambda checked=False, i=index: self._switch_section(i))
        self.nav_buttons.append(button)

        sidebar = self.findChild(QWidget, "settingsSidebar")
        if sidebar and sidebar.layout():
            sidebar.layout().addWidget(button)
        return index

    def _nav_icon(self, kind: str) -> QIcon:
        pix = QPixmap(20, 20)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(160, 205, 235), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        c = QPointF(10, 10)
        if kind == "general":
            p.drawEllipse(QRectF(5, 5, 10, 10))
            p.drawLine(QPointF(10, 3), QPointF(10, 6))
            p.drawLine(QPointF(10, 14), QPointF(10, 17))
            p.drawLine(QPointF(3, 10), QPointF(6, 10))
            p.drawLine(QPointF(14, 10), QPointF(17, 10))
        elif kind == "llm":
            p.drawEllipse(QRectF(4, 5, 12, 10))
            p.drawLine(QPointF(8, 15), QPointF(12, 15))
        elif kind == "whisper":
            p.drawRoundedRect(QRectF(7, 4, 6, 9), 3, 3)
            p.drawLine(QPointF(5, 10), QPointF(5, 12))
            p.drawLine(QPointF(15, 10), QPointF(15, 12))
            p.drawLine(QPointF(10, 13), QPointF(10, 17))
            p.drawLine(QPointF(7, 17), QPointF(13, 17))
        elif kind == "audio":
            for x, h in [(5, 5), (8, 10), (11, 14), (14, 8)]:
                p.drawLine(QPointF(x, 10 - h / 2), QPointF(x, 10 + h / 2))
        elif kind == "live":
            p.setBrush(QColor(160, 205, 235))
            p.drawEllipse(QRectF(7, 7, 6, 6))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QRectF(4, 4, 12, 12))
        elif kind == "advanced":
            for y, x in [(6, 8), (10, 12), (14, 7)]:
                p.drawLine(QPointF(4, y), QPointF(16, y))
                p.setBrush(QColor(160, 205, 235))
                p.drawEllipse(QRectF(x - 2, y - 2, 4, 4))
                p.setBrush(Qt.NoBrush)
        elif kind == "diagnostics":
            p.drawLine(QPointF(5, 10), QPointF(8, 13))
            p.drawLine(QPointF(8, 13), QPointF(15, 6))
        else:
            p.drawRect(QRectF(5, 5, 10, 12))
            p.drawLine(QPointF(8, 8), QPointF(12, 8))
            p.drawLine(QPointF(8, 11), QPointF(12, 11))
        p.end()
        return QIcon(pix)

    def _switch_section(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
            button.setProperty("active", i == index)
            button.style().unpolish(button)
            button.style().polish(button)

    def _section_page(self, title: str, description: str) -> QVBoxLayout:
        page = QWidget()
        self._page_widgets.append(page)
        page.setObjectName("settingsPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 2, 12, 4)
        layout.setSpacing(12)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 4)
        header_layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setObjectName("settingsPageTitle")
        desc_label = QLabel(description)
        desc_label.setObjectName("settingsPageDescription")
        desc_label.setWordWrap(True)
        header_layout.addWidget(title_label)
        header_layout.addWidget(desc_label)
        layout.addWidget(header)
        return layout

    def _group(self, title: str, rows):
        group = QWidget()
        group.setObjectName("settingsGroup")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        label = QLabel(title)
        label.setObjectName("settingsGroupTitle")
        layout.addWidget(label)
        for row in rows:
            layout.addWidget(row)
        return group

    def _setting_row(self, title: str, hint: str, control):
        row = QWidget()
        row.setObjectName("settingsRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(18)

        label_box = QVBoxLayout()
        label_box.setContentsMargins(0, 0, 0, 0)
        label_box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("settingsRowTitle")
        title_label.setWordWrap(True)
        hint_label = QLabel(hint)
        hint_label.setObjectName("settingsRowHint")
        hint_label.setWordWrap(True)
        hint_label.setMinimumWidth(0)
        label_box.addWidget(title_label)
        label_box.addWidget(hint_label)
        layout.addLayout(label_box, 1)

        control_widget = self._layout_widget(control) if isinstance(control, QHBoxLayout) else control
        control_widget.setObjectName(control_widget.objectName() or "settingsControl")
        fixed_width = control_widget.property("fixedControlWidth")
        if isinstance(fixed_width, int) and fixed_width > 0:
            control_widget.setFixedWidth(fixed_width)
        elif control_widget.objectName() == "settingsPill":
            pass
        elif control_widget.objectName() == "settingsSwitchControl":
            control_widget.setFixedWidth(300)
        else:
            control_widget.setMinimumWidth(220)
            control_widget.setMaximumWidth(430)
        layout.addWidget(control_widget, 0, Qt.AlignVCenter)
        return row

    def _layout_widget(self, source_layout: QHBoxLayout):
        box = QWidget()
        box.setObjectName("settingsInlineControl")
        box.setLayout(source_layout)
        return box

    def _switch_control(self, switch: ToggleSwitch):
        box = QWidget()
        box.setObjectName("settingsSwitchControl")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(switch, 0, Qt.AlignLeft | Qt.AlignVCenter)
        layout.addStretch(1)
        return box

    def _set_control_width(self, widget: QWidget, width: int):
        widget.setProperty("fixedControlWidth", width)
        widget.setMinimumWidth(width)
        widget.setMaximumWidth(width)

    def _info_card(self, title: str, text: str):
        card = QWidget()
        card.setObjectName("settingsInfoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("settingsInfoTitle")
        text_label = QLabel(text)
        text_label.setObjectName("settingsInfoText")
        text_label.setWordWrap(True)
        text_label.setMinimumWidth(0)
        text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        layout.addWidget(title_label)
        layout.addWidget(text_label)
        return card

    def _status_card(self, title: str, value: str, badge: str):
        card = QWidget()
        card.setObjectName("settingsStatusCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)
        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("settingsInfoTitle")
        value_label = QLabel(value)
        value_label.setObjectName("settingsInfoText")
        value_label.setWordWrap(True)
        text_box.addWidget(title_label)
        text_box.addWidget(value_label)
        layout.addLayout(text_box, 1)
        layout.addWidget(self._pill(badge))
        return card

    def _pill(self, text: str):
        metrics = QFontMetrics(QFont("Roboto, Segoe UI, Arial", 12))
        width = max(96, min(metrics.horizontalAdvance(text) + 36, 260))
        shown = metrics.elidedText(text, Qt.ElideMiddle, width - 28)
        label = QLabel(shown)
        label.setObjectName("settingsPill")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(30)
        label.setMinimumWidth(width)
        label.setMaximumWidth(width)
        label.setToolTip(text)
        return label

    def _open_config_folder(self):
        try:
            os.startfile(str(CONFIG_PATH.parent))
        except Exception as exc:
            self.show_result_dialog("Не удалось открыть папку", [("error", str(exc))], "error")

    def _settings_stylesheet(self):
        return """
            QDialog {
                background: transparent;
                color: #e5e7eb;
                font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QWidget#settingsShell {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #07101f, stop:0.55 #091426, stop:1 #11112c);
                border: 1px solid rgba(56, 189, 248, 0.14);
                border-radius: 24px;
            }
            QWidget#settingsBody,
            QWidget#settingsPage,
            QStackedWidget#settingsStack,
            QScrollArea#settingsPageScroll,
            QScrollArea#settingsPageScroll > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QWidget#settingsSidebar {
                background: rgba(9, 19, 36, 0.48);
                border: none;
                border-radius: 18px;
            }
            QLabel#settingsBrandTitle {
                color: #ffffff;
                font-size: 18px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#settingsBrandSubtitle {
                color: #8ea7c7;
                font-size: 12px;
                background: transparent;
            }
            QPushButton#settingsNavItem {
                min-height: 42px;
                text-align: left;
                padding: 0 12px 0 10px;
                color: #9fb4d0;
                background: transparent;
                border: none;
                border-radius: 12px;
                font-weight: 600;
            }
            QPushButton#settingsNavItem:hover {
                color: #e5e7eb;
                background: rgba(30, 48, 80, 0.46);
            }
            QPushButton#settingsNavItem:checked,
            QPushButton#settingsNavItem[active="true"] {
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(14, 116, 144, 0.82),
                    stop:1 rgba(109, 40, 217, 0.70));
                border: none;
            }
            QLabel#settingsPageTitle {
                color: #ffffff;
                font-size: 27px;
                font-weight: 750;
                background: transparent;
            }
            QLabel#settingsPageDescription {
                color: #9db4d3;
                font-size: 13px;
                background: transparent;
            }
            QWidget#settingsFooter {
                background: rgba(13, 23, 41, 0.58);
                border: none;
                border-radius: 18px;
            }
            QLabel#settingsTitle {
                color: #ffffff;
                font-size: 26px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#settingsSubtitle {
                color: #b7c8df;
                font-size: 14px;
                background: transparent;
            }
            QLabel#settingsPath {
                color: #7488a8;
                font-size: 11px;
                background: transparent;
            }
            QLabel { color: #cfe4ff; background: transparent; }
            QWidget#settingsToggleRow {
                background: transparent;
                border: none;
            }
            QLabel#settingsToggleText {
                color: #d7e7fb;
                font-size: 13px;
                background: transparent;
            }
            QWidget#settingsInlineControl {
                background: transparent;
                border: none;
            }
            QWidget#settingsGroup {
                background: rgba(15, 27, 48, 0.54);
                border: 1px solid rgba(148, 163, 184, 0.08);
                border-radius: 18px;
            }
            QLabel#settingsGroupTitle {
                color: #eef6ff;
                font-size: 14px;
                font-weight: 700;
                background: transparent;
            }
            QWidget#settingsRow {
                background: transparent;
                border: none;
            }
            QLabel#settingsRowTitle {
                color: #dbeafe;
                font-size: 13px;
                font-weight: 650;
                background: transparent;
            }
            QLabel#settingsRowHint {
                color: #7f96b7;
                font-size: 11px;
                background: transparent;
            }
            QWidget#settingsInfoCard,
            QWidget#settingsStatusCard {
                background: rgba(15, 27, 48, 0.50);
                border: 1px solid rgba(56, 189, 248, 0.10);
                border-radius: 17px;
            }
            QLabel#settingsInfoTitle {
                color: #eef6ff;
                font-size: 14px;
                font-weight: 700;
                background: transparent;
            }
            QLabel#settingsInfoText {
                color: #aec1dc;
                font-size: 12px;
                background: transparent;
            }
            QLabel#settingsInlineStatus {
                color: #93c5fd;
                font-size: 12px;
                background: transparent;
                padding: 2px 4px;
            }
            QLabel#settingsPill {
                color: #cfefff;
                background: rgba(56, 189, 248, 0.12);
                border: 1px solid rgba(56, 189, 248, 0.20);
                border-radius: 15px;
                padding: 4px 10px;
                font-weight: 650;
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: transparent;
                border: none;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 38px;
                background: rgba(18, 30, 52, 0.72);
                color: #e5e7eb;
                border: 1px solid rgba(148, 163, 184, 0.10);
                border-radius: 14px;
                padding: 7px 12px;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                background: rgba(22, 38, 66, 0.80);
                border: 1px solid rgba(56, 189, 248, 0.22);
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid rgba(56, 189, 248, 0.48);
                background: rgba(23, 42, 72, 0.88);
            }
            QComboBox::drop-down {
                width: 30px;
                border: none;
                border-left: none;
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background: rgba(9, 18, 34, 0.98);
                color: #e5e7eb;
                border: 1px solid rgba(56, 189, 248, 0.24);
                border-radius: 14px;
                padding: 8px;
                outline: 0;
                selection-background-color: rgba(37, 99, 235, 0.52);
                selection-color: #ffffff;
            }
            QComboBox QAbstractItemView::item {
                min-height: 30px;
                padding: 7px 10px;
                border-radius: 7px;
            }
            QComboBox QAbstractItemView::item:hover {
                background: rgba(30, 64, 111, 0.68);
                color: #ffffff;
            }
            QComboBox QAbstractItemView::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0e7490, stop:1 #5b21b6);
                color: #ffffff;
            }
            QPushButton {
                min-height: 38px;
                color: #e5e7eb;
                background: rgba(26, 42, 68, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 15px;
                padding: 0 18px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: rgba(37, 57, 90, 0.82);
                border-color: rgba(56, 189, 248, 0.28);
            }
            QPushButton[variant="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(14, 116, 144, 0.92),
                    stop:1 rgba(109, 40, 217, 0.90));
                border-color: rgba(103, 232, 249, 0.30);
                color: #ffffff;
            }
            QPushButton[variant="primary"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(8, 145, 178, 0.98),
                    stop:1 rgba(124, 58, 237, 0.96));
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 0px;
                border: none;
                background: transparent;
            }
            QScrollArea QScrollBar:vertical, QComboBox QScrollBar:vertical, QScrollBar:vertical {
                background: #0a1322;
                width: 12px;
                margin: 4px 2px 4px 2px;
                border-radius: 7px;
            }
            QScrollArea QScrollBar::handle:vertical, QComboBox QScrollBar::handle:vertical, QScrollBar::handle:vertical {
                background: #31547d;
                min-height: 36px;
                border-radius: 6px;
            }
            QScrollArea QScrollBar::handle:vertical:hover, QComboBox QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }
            QScrollArea QScrollBar::add-page:vertical, QScrollArea QScrollBar::sub-page:vertical,
            QComboBox QScrollBar::add-page:vertical, QComboBox QScrollBar::sub-page:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: #0a1322;
                border-radius: 7px;
            }
            QScrollArea QScrollBar::add-line:vertical, QScrollArea QScrollBar::sub-line:vertical,
            QComboBox QScrollBar::add-line:vertical, QComboBox QScrollBar::sub-line:vertical,
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }
            QScrollArea QScrollBar:horizontal, QComboBox QScrollBar:horizontal, QScrollBar:horizontal {
                background: #0a1322;
                height: 12px;
                border-radius: 6px;
            }
            QScrollArea QScrollBar::handle:horizontal, QComboBox QScrollBar::handle:horizontal, QScrollBar::handle:horizontal {
                background: #31547d;
                min-width: 36px;
                border-radius: 5px;
            }
            QToolTip {
                background: #0b1424;
                color: #e5e7eb;
                border: 1px solid #38bdf8;
                border-radius: 8px;
                padding: 6px;
            }
        """

    def _elide_path(self, value: str) -> str:
        metrics = QFontMetrics(QFont("Segoe UI", 10))
        return metrics.elidedText(value, Qt.ElideMiddle, 620)

    def _line(self, value: str, password=False):
        widget = QLineEdit(value)
        widget.setMinimumHeight(self.FIELD_HEIGHT)
        widget.setMinimumWidth(360)
        widget.setClearButtonEnabled(not password)
        if password:
            widget.setEchoMode(QLineEdit.Password)
        if value and not password:
            widget.setToolTip(value)
            widget.setCursorPosition(0)
        return widget

    def _combo(self, items, current, minimum_width=260, editable=False):
        combo = SoftComboBox()
        combo.setEditable(editable)
        combo.setMinimumHeight(self.FIELD_HEIGHT)
        combo.setMinimumWidth(minimum_width)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(24)
        combo.setMaxVisibleItems(14)
        for item in items:
            combo.addItem(str(item), item)
        if current is not None:
            combo.setCurrentText(str(current))
        combo.currentTextChanged.connect(lambda text, c=combo: c.setToolTip(text))
        combo.setToolTip(combo.currentText())
        return combo

    def _toggle_row(self, label: str, checked=False):
        row = QWidget()
        row.setObjectName("settingsToggleRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)
        toggle = ToggleSwitch(bool(checked))
        text = QLabel(label)
        text.setObjectName("settingsToggleText")
        text.setWordWrap(True)
        layout.addWidget(toggle, 0, Qt.AlignVCenter)
        layout.addWidget(text, 1, Qt.AlignVCenter)
        return toggle, row

    def _spin(self, minimum, maximum, value):
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setButtonSymbols(QSpinBox.NoButtons)
        widget.setMinimumHeight(self.FIELD_HEIGHT)
        widget.setMinimumWidth(180)
        return widget

    def _double_spin(self, minimum, maximum, value, step):
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(2)
        widget.setValue(value)
        widget.setButtonSymbols(QDoubleSpinBox.NoButtons)
        widget.setMinimumHeight(self.FIELD_HEIGHT)
        widget.setMinimumWidth(180)
        return widget

    def _button(self, text: str, variant="secondary"):
        btn = QPushButton(text)
        btn.setProperty("variant", variant)
        btn.setMinimumHeight(self.FIELD_HEIGHT)
        btn.setCursor(QCursor(Qt.PointingHandCursor))
        return btn

    def _mini_field(self, label: str, widget: QWidget):
        widget.setMinimumWidth(70)
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        caption = QLabel(label)
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #8fa8c7; font-size: 11px; background: transparent;")
        layout.addWidget(caption)
        layout.addWidget(widget)
        return box

    def _refresh_combo_metrics(self):
        for combo in self.findChildren(QComboBox):
            for i in range(combo.count()):
                text = combo.itemText(i)
                combo.setItemData(i, text, Qt.ToolTipRole)
            if hasattr(combo, "refresh_popup_metrics"):
                combo.refresh_popup_metrics()

    def _fill_audio_devices(self, settings: dict):
        try:
            devices = sd.query_devices()
            for index, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0)) > 0:
                    self.input_device.addItem(f"{index}: {dev['name']}", index)
                    if settings.get("input_device") == index:
                        self.input_device.setCurrentIndex(self.input_device.count() - 1)
                if int(dev.get("max_output_channels", 0)) > 0:
                    self.system_audio_device.addItem(f"{index}: {dev['name']}", index)
                    if settings.get("system_audio_device") == index:
                        self.system_audio_device.setCurrentIndex(self.system_audio_device.count() - 1)
        except Exception:
            pass
        if pyaudio is not None:
            try:
                pa = pyaudio.PyAudio()
                try:
                    for item in pa.get_loopback_device_info_generator():
                        index = int(item["index"])
                        self.system_audio_device.addItem(f"{index}: {item['name']} (loopback)", index)
                        if settings.get("system_audio_device") == index:
                            self.system_audio_device.setCurrentIndex(self.system_audio_device.count() - 1)
                finally:
                    pa.terminate()
            except Exception:
                pass
        self._refresh_combo_metrics()

    def _collect(self) -> dict:
        api_key = self.api_key.text().strip()
        if not api_key and self._initial_api_key:
            api_key = self._initial_api_key
        return normalize_settings({
            "whisper_url": self.whisper_url.text().strip(),
            "llm_url": self.llm_url.text().strip(),
            "llm_api_type": self.llm_api_type.currentText().strip().lower(),
            "llm_model": self.llm_model.currentText().strip(),
            "openai_api_key": api_key,
            "openwebui_chat_id": self.chat_id.text().strip(),
            "hotkey": self.hotkey.text().strip() or "ctrl+alt+a",
            "chunk_seconds": self.chunk_seconds.value(),
            "buffer_minutes": self.buffer_minutes.value(),
            "language": self.language.text().strip() or "ru",
            "input_device": self.input_device.currentData(),
            "prompt_mode": self.prompt_mode.currentText().strip().lower(),
            "last_lines": self.last_lines.value(),
            "llm_timeout": self.llm_timeout.value(),
            "temperature": self.temperature.value(),
            "max_tokens": self.max_tokens.value(),
            "system_audio_enabled": self.system_audio_enabled.isChecked(),
            "system_audio_device": self.system_audio_device.currentData(),
            "auto_system_loopback": self.auto_system_loopback.isChecked(),
            "mix_mic_and_system_audio": self.mix_audio.isChecked(),
            "live_transcription": self.live_transcription.isChecked(),
            "live_chunk_seconds": self.live_chunk_seconds.value(),
            "save_history": self.save_history.isChecked(),
            "diagnostic_mode": self.diagnostic_mode.isChecked(),
        })

    def _delete_local_history(self):
        confirmed = self.show_result_dialog(
            "Удалить локальную историю?",
            [("warn", "Будут удалены сохранённые расшифровки, ответы и старый приватный debug-log. Отменить действие после подтверждения нельзя.")],
            "warn",
            confirm_text="Удалить",
        )
        if not confirmed:
            return
        removed = delete_local_history()
        self.show_result_dialog(
            "Локальная история",
            [("ok", f"Удалено файлов: {removed}." if removed else "Сохранённая история не найдена.")],
            "ok",
        )

    def restore_defaults(self):
        self.settings_result = normalize_settings(DEFAULT_SETTINGS)
        save_config(self.settings_result)
        self.show_result_dialog("PySufler", [("ok", "Настройки сброшены к безопасным значениям по умолчанию.")], "ok")
        self.accept()

    def refresh_models(self):
        current = self.llm_model.currentText().strip()
        self._set_model_status("Обновляю список моделей...", "info")
        try:
            models = self._request_models()
        except Exception as exc:
            self._set_model_status(f"Не удалось получить модели: {exc}", "error")
            return
        self.llm_model.clear()
        if current and current not in models:
            self.llm_model.addItem(current)
        for model in models:
            self.llm_model.addItem(model)
        if current:
            self.llm_model.setCurrentText(current)
            if current not in models:
                self._set_model_status("Список обновлён. Текущая модель сохранена, но её нет в ответе API.", "warn")
            else:
                self._set_model_status(f"Список обновлён. Моделей найдено: {len(models)}", "ok")
        else:
            self._set_model_status(f"Список обновлён. Моделей найдено: {len(models)}", "ok")
        self._refresh_combo_metrics()

    def _set_model_status(self, text: str, level="info"):
        if not hasattr(self, "model_status"):
            return
        colors = {"ok": "#5eead4", "warn": "#fbbf24", "error": "#fca5a5", "info": "#93c5fd"}
        self.model_status.setText(text)
        self.model_status.setStyleSheet(f"color: {colors.get(level, '#93c5fd')}; background: transparent;")

    def _request_models(self):
        settings = self._collect()
        url = settings["llm_url"].rstrip("/")
        if settings["llm_api_type"] in ("openwebui", "openai"):
            if "/api/chat/completions" in url:
                models_url = url.replace("/api/chat/completions", "/api/models")
            elif "/v1/chat/completions" in url:
                models_url = url.replace("/v1/chat/completions", "/v1/models")
            else:
                models_url = url.rsplit("/", 1)[0] + "/models"
        elif settings["llm_api_type"] == "ollama":
            models_url = url.replace("/api/chat", "/api/tags")
        else:
            models_url = url.replace("/api/generate", "/api/tags")
        headers = {}
        if settings["openai_api_key"]:
            headers["Authorization"] = f"Bearer {settings['openai_api_key']}"
        response = requests.get(models_url, headers=headers, timeout=min(settings["llm_timeout"], 30))
        response.raise_for_status()
        data = response.json()
        if "data" in data:
            return [item.get("id") for item in data["data"] if item.get("id")]
        if "models" in data:
            return [item.get("name") or item.get("model") for item in data["models"] if item.get("name") or item.get("model")]
        return []

    def check_connection(self):
        lines = []
        target_model = self.llm_model.currentText().strip() or "qwen3.5:9b"
        try:
            models = self._request_models()
            lines.append(("ok", f"LLM — доступен, моделей найдено: {len(models)}"))
            if target_model in models:
                lines.append(("ok", f"Модель {target_model} — найдена"))
            else:
                lines.append(("warn", f"Модель {target_model} — не найдена в списке API, текущее значение сохранено"))
        except Exception as exc:
            lines.append(("error", f"LLM — ошибка: {exc}"))
            lines.append(("warn", f"Модель {target_model} — не проверена"))
        try:
            response = requests.get(self.whisper_url.text().strip(), timeout=5)
            status = "ok" if response.status_code < 500 else "warn"
            lines.append((status, f"Whisper — endpoint HTTP {response.status_code}"))
        except Exception as exc:
            lines.append(("error", f"Whisper — ошибка: {exc}"))
        try:
            if self.system_audio_enabled.isChecked():
                device = get_system_audio_device(self._collect())
                name = device.get("name", "WASAPI loopback") if isinstance(device, dict) else str(device)
                lines.append(("ok", f"Системный звук — доступен: {name}"))
            else:
                lines.append(("warn", "Системный звук — выключен в настройках"))
        except Exception as exc:
            lines.append(("error", f"Системный звук — ошибка: {exc}"))
        level = "error" if any(item[0] == "error" for item in lines) else "ok"
        self.show_result_dialog("Проверка подключения", lines, level)

    def accept(self):
        if self.settings_result is None:
            self.settings_result = self._collect()
            try:
                save_config(self.settings_result)
            except OSError:
                self.settings_result = None
                self.show_result_dialog("Не удалось сохранить настройки", [("error", "Проверьте доступ к локальной папке и Windows DPAPI. Старые настройки сохранены.")], "error")
                return
        super().accept()


class MainScrollArea(QScrollArea):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        content = self.widget()
        if content is not None and content.layout() is not None:
            content.resize(self.viewport().width(), max(
                self.viewport().height(), content.layout().minimumSize().height()
            ))


class ModernMeetingHelperGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                            if env_bool("GUI_TOPMOST", False)
                            else Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setWindowTitle(APP_NAME)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect(0, 0, 1200, 950)
        start_w = min(1200, max(680, available.width() - 40))
        start_h = min(950, max(480, available.height() - 40))
        self.resize(start_w, start_h)
        self.setMinimumSize(680, 480)

        self.settings = load_settings()

        self.stream = None
        self.system_stream = None
        self.system_pa = None
        self.frames = []
        self.system_frames = []
        self.full_frames = []
        self.full_system_frames = []
        self.frames_lock = threading.Lock()
        self.system_frames_lock = threading.Lock()
        self.is_recording = False
        self.system_audio_active = False
        self.system_audio_seen = False
        self.system_audio_has_signal = False
        self._system_status_reported = False
        self.recording_started_at = None
        self.audio_level = 0.0
        self._drag_pos = None
        self._resize_margin = 10
        self._resize_edges = ""
        self._resize_start_global = None
        self._resize_start_geom = None
        self._last_live_text = ""
        self._llm_checked = False
        self.conversation_history = ConversationHistory(max_exchanges=5)

        self._build_ui()
        self._connect_bridge()
        self.set_status("Готов")

        self._timer_tick = QTimer(self)
        self._timer_tick.timeout.connect(self._update_timer)
        self._timer_tick.start(250)
        self._btn_phase = 0.0

    # ── Bridge ────────────────────────────────────────────────────────────────

    def _connect_bridge(self):
        _bridge.set_status_sig.connect(self.set_status)
        _bridge.set_question_sig.connect(self._set_question_text)
        _bridge.set_answer_sig.connect(self._set_answer_text)
        _bridge.append_question_sig.connect(self._append_question_text)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Animated background (behind everything)
        self._bg = NeonBackground(self)
        self._bg.setGeometry(1, 1, self.width() - 2, self.height() - 2)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 16)
        outer.setSpacing(0)

        content = QWidget()
        content.setObjectName("content")
        content.setStyleSheet("QWidget#content { background: transparent; }")
        scroll = MainScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
            QScrollBar:vertical { background: transparent; width: 6px; margin: 4px 0; }
            QScrollBar::handle:vertical { background: #304c6b; min-height: 28px; border-radius: 3px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(22, 20, 22, 16)
        vbox.setSpacing(8)

        self._build_header(vbox)
        self._build_controls(vbox)
        self._build_question_card(vbox)
        self._build_waveform(vbox)
        self._build_answer_card(vbox)
        self._build_status_bar(vbox)

    def _label(self, text, size=13, bold=False, color=None):
        lbl = QLabel(text)
        w = QFont.Bold if bold else QFont.Normal
        lbl.setFont(QFont(system_font_family(), size, w))
        col = color or TEXT
        lbl.setStyleSheet(f"color: {col.name()}; background: transparent;")
        return lbl

    def _build_header(self, vbox):
        row = QHBoxLayout()
        row.setSpacing(16)

        ring = RingIcon()
        row.addWidget(ring)

        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = self._label(APP_NAME, 24, True)
        title.setWordWrap(True)
        title_col.addWidget(title)
        subtitle = self._label("Суфлёр совещаний · локальный помощник", 11, color=QColor("#8aabcc"))
        subtitle.setWordWrap(True)
        title_col.addWidget(subtitle)
        self.model_label = self._label(f"Модель: {self.settings['llm_model']}", 10,
                                       color=QColor("#6f8fb1"))
        title_col.addWidget(self.model_label)
        row.addLayout(title_col, 1)

        self.settings_btn = NeonButton("", QColor(70, 110, 155), QColor(10, 18, 36), icon_kind="settings")
        self.settings_btn.setAccessibleName("Настройки")
        self.settings_btn.setToolTip("Настройки")
        self.settings_btn.set_size(42, 30)
        self.settings_btn.clicked.connect(self.open_settings)
        row.addWidget(self.settings_btn)

        # Window controls
        for char, hover_c, cmd in (("—", "#152545", self._minimize),
                                    ("×", "#3a1636", self.close)):
            btn = NeonButton(char, QColor(50, 80, 140), QColor(10, 18, 36))
            btn.set_size(34, 30)
            btn.clicked.connect(cmd)
            row.addWidget(btn)

        vbox.addLayout(row)

    def _build_controls(self, vbox):
        row = QHBoxLayout()
        row.setSpacing(10)

        self.listen_btn = NeonButton("Запись", CYAN, CYAN_DIM, icon_kind="record")
        self.listen_btn.setMinimumSize(130, 48)
        self.listen_btn.setMaximumSize(16777215, 48)
        self.listen_btn.clicked.connect(self.toggle_recording)
        row.addWidget(self.listen_btn)

        self.answer_btn = NeonButton("Ответ", PURPLE, PURPLE_DIM, icon_kind="answer")
        self.answer_btn.setMinimumSize(120, 48)
        self.answer_btn.setMaximumSize(16777215, 48)
        self.answer_btn.clicked.connect(self.request_answer)
        row.addWidget(self.answer_btn)

        self.clear_btn = NeonButton("Очистить", QColor(58, 80, 112), QColor(16, 28, 50), icon_kind="clear")
        self.clear_btn.setMinimumSize(120, 48)
        self.clear_btn.setMaximumSize(16777215, 48)
        self.clear_btn.clicked.connect(self.clear_fields)
        row.addWidget(self.clear_btn)

        self.check_btn = NeonButton("Проверить", QColor(68, 96, 130), QColor(14, 26, 46), icon_kind="check")
        self.check_btn.setMinimumSize(130, 48)
        self.check_btn.setMaximumSize(16777215, 48)
        self.check_btn.clicked.connect(self.run_diagnostics)
        row.addWidget(self.check_btn)

        row.addStretch()

        # Toggle
        toggle_col = QVBoxLayout()
        toggle_col.setSpacing(2)
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(10)
        self.toggle = ToggleSwitch(self.settings["system_audio_enabled"])
        self.toggle.toggled.connect(self._on_toggle)
        toggle_row.addWidget(self.toggle)
        toggle_row.addWidget(self._label("Слушать собеседника", 12))
        toggle_col.addLayout(toggle_row)
        toggle_col.addWidget(self._label("микрофон + звук системы", 10, color=MUTED))
        vbox.addLayout(row)
        vbox.addLayout(toggle_col)

    def _build_config_bar(self, vbox):
        label = self._label(f"Модель: {self.settings['llm_model']}", 11,
                            color=QColor("#7fa7c8"))
        label.setWordWrap(True)
        vbox.addWidget(label)

    def _build_question_card(self, vbox):
        sources = QHBoxLayout()
        sources.setSpacing(18)
        self.mic_indicator = self._label("MIC · Выключен", 11, color=MUTED)
        self.sys_indicator = self._label("SYS · Выключен", 11, color=MUTED)
        sources.addWidget(self.mic_indicator)
        sources.addWidget(self.sys_indicator)
        sources.addStretch()
        vbox.addLayout(sources)
        card = GlassCard(CYAN)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 18, 28, 18)
        cl.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(self._label("Распознанный вопрос", 14, True))
        header.addStretch()
        copy_q = CopyButton()
        copy_q.clicked.connect(self.copy_question)
        header.addWidget(copy_q)
        cl.addLayout(header)

        self.question_box = NeonTextEdit()
        self.question_box.setPlaceholderText("Распознанный текст появится здесь после записи")
        self.question_box.setMinimumHeight(88)
        self.question_box.setViewportMargins(0, 0, 0, 28)
        cl.addWidget(self.question_box, 1)
        card.setMinimumHeight(162)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vbox.addWidget(card, 2)

    def _build_waveform(self, vbox):
        self.waveform = NeonWaveform(lambda: self.is_recording,
                                     lambda: self.audio_level)
        self.waveform.setMaximumHeight(56)
        vbox.addWidget(self.waveform)

    def _build_answer_card(self, vbox):
        card = GlassCard(PURPLE)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 18, 28, 20)
        cl.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(self._label("Ответ помощника", 14, True))
        header.addStretch()
        copy_a = CopyButton()
        copy_a.clicked.connect(self.copy_answer)
        header.addWidget(copy_a)
        cl.addLayout(header)

        self.answer_box = NeonTextEdit()
        self.answer_box.setPlaceholderText("Ответ появится здесь после обработки вопроса")
        self.answer_box.setMinimumHeight(155)
        cl.addWidget(self.answer_box, 1)
        card.setMinimumHeight(215)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vbox.addWidget(card, 7)

    def _build_status_bar(self, vbox):
        bar = GlassStatusBar()
        bar.setFixedHeight(44)
        bar.setStyleSheet("background: transparent;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 0, 16, 0)
        bl.setSpacing(10)

        self.status_dot = StatusDot()
        bl.addWidget(self.status_dot)

        self.status_lbl = self._label("", 13, True)
        bl.addWidget(self.status_lbl)

        bl.addStretch()

        self.mode_lbl = self._label("", 13, color=MUTED)
        bl.addWidget(self.mode_lbl)
        self._refresh_mode_label()

        bl.addStretch()

        self.timer_lbl = self._label("00:00", 13, True, CYAN)
        bl.addWidget(self.timer_lbl)

        grip = QSizeGrip(bar)
        grip.setFixedSize(18, 18)
        grip.setStyleSheet("background: transparent;")
        bl.addWidget(grip)

        vbox.addSpacing(2)
        vbox.addWidget(bar, 0)

    # ── Window chrome ─────────────────────────────────────────────────────────

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._bg.setGeometry(1, 1, max(0, self.width() - 2), max(0, self.height() - 2))

    def _hit_resize_edges(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self._resize_margin
        edges = ""
        if x <= m:
            edges += "l"
        elif x >= w - m:
            edges += "r"
        if y <= m:
            edges += "t"
        elif y >= h - m:
            edges += "b"
        return edges

    def _update_resize_cursor(self, edges):
        if edges in ("lt", "rb"):
            self.setCursor(Qt.SizeFDiagCursor)
        elif edges in ("rt", "lb"):
            self.setCursor(Qt.SizeBDiagCursor)
        elif edges in ("l", "r"):
            self.setCursor(Qt.SizeHorCursor)
        elif edges in ("t", "b"):
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _apply_resize(self, global_pos):
        if not self._resize_edges or self._resize_start_global is None:
            return
        delta = global_pos - self._resize_start_global
        geom = QRect(self._resize_start_geom)
        min_w = self.minimumWidth()
        min_h = self.minimumHeight()

        if "l" in self._resize_edges:
            new_left = geom.left() + delta.x()
            if geom.right() - new_left + 1 >= min_w:
                geom.setLeft(new_left)
        if "r" in self._resize_edges:
            new_right = geom.right() + delta.x()
            if new_right - geom.left() + 1 >= min_w:
                geom.setRight(new_right)
        if "t" in self._resize_edges:
            new_top = geom.top() + delta.y()
            if geom.bottom() - new_top + 1 >= min_h:
                geom.setTop(new_top)
        if "b" in self._resize_edges:
            new_bottom = geom.bottom() + delta.y()
            if new_bottom - geom.top() + 1 >= min_h:
                geom.setBottom(new_bottom)

        self.setGeometry(geom)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        shell_path = QPainterPath()
        shell_path.addRoundedRect(QRectF(1, 1, w - 2, h - 2), 30, 30)
        p.setClipPath(shell_path)

        # Main shell
        shell_grad = QLinearGradient(0, 0, 0, h)
        shell_grad.setColorAt(0.0, QColor(7, 12, 26, 238))
        shell_grad.setColorAt(0.55, QColor(8, 16, 32, 232))
        shell_grad.setColorAt(1.0, QColor(5, 10, 22, 236))
        p.setBrush(shell_grad)
        p.setPen(Qt.NoPen)
        p.drawPath(shell_path)

        p.setClipping(False)
        glow = QLinearGradient(0, 0, w, h)
        glow.setColorAt(0.0, QColor(35, 190, 255, 165))
        glow.setColorAt(0.45, QColor(68, 120, 255, 90))
        glow.setColorAt(1.0, QColor(170, 80, 255, 145))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QBrush(glow), 1.1))
        p.drawRoundedRect(QRectF(1.5, 1.5, w - 3, h - 3), 30, 30)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            edges = self._hit_resize_edges(e.position().toPoint())
            if edges:
                self._resize_edges = edges
                self._resize_start_global = e.globalPosition().toPoint()
                self._resize_start_geom = self.geometry()
                return
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._resize_edges and e.buttons() & Qt.LeftButton:
            self._apply_resize(e.globalPosition().toPoint())
            return
        if self._drag_pos and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            return
        self._update_resize_cursor(self._hit_resize_edges(e.position().toPoint()))

    def mouseReleaseEvent(self, _):
        self._drag_pos = None
        self._resize_edges = ""
        self._resize_start_global = None
        self._resize_start_geom = None

    def _minimize(self):
        self.showMinimized()

    def closeEvent(self, e):
        self.on_close()
        e.accept()

    # ── Status ────────────────────────────────────────────────────────────────

    def set_status(self, text: str):
        self.status_lbl.setToolTip(text)
        self.status_lbl.setText(text)
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setMinimumWidth(0)
        if text.startswith("Ошибка"):
            self.status_dot.set_color(RED)
        elif text in (
            "Слушаю...",
            "Распознаю...",
            "Запрашиваю ответ...",
            "Распознавание в реальном времени...",
            "Промежуточный текст",
        ):
            self.status_dot.set_color(CYAN)
        elif text in ("Ничего не распознано", "Запись остановлена"):
            self.status_dot.set_color(MUTED)
        else:
            self.status_dot.set_color(GREEN)

    def _append_question_text(self, text: str):
        self.question_box.setPlainText(text)
        cursor = self.question_box.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.question_box.setTextCursor(cursor)
        self.question_box.ensureCursorVisible()

    def set_error(self, text: str):
        self.set_status(f"Ошибка: {text[:90]}")

    def _update_timer(self):
        self._refresh_source_indicators()
        self._btn_phase = (self._btn_phase + 0.12) % (math.pi * 2)
        if self.is_recording and self.recording_started_at is not None:
            elapsed = int(time.time() - self.recording_started_at)
            self.timer_lbl.setText(f"{elapsed // 60:02d}:{elapsed % 60:02d}")
            pulse = 0.5 + 0.5 * math.sin(self._btn_phase)
            r = int(68 + pulse * 28)
            g = int(16 + pulse * 8)
            b = int(36 + pulse * 14)
            self.listen_btn.configure(
                fg_color=QColor(r, g, b),
                border_color=RED,
            )
        else:
            self.timer_lbl.setText("00:00")
            self.audio_level = max(0.0, self.audio_level * 0.88)

    def _on_toggle(self, v: bool):
        self.settings["system_audio_enabled"] = v
        save_config(self.settings)
        self._refresh_mode_label()

    def _refresh_mode_label(self):
        if not self.settings["system_audio_enabled"]:
            mode = "микрофон"
        elif self.settings.get("mix_mic_and_system_audio", True):
            mode = "микрофон + звук системы"
        else:
            mode = "звук системы"
        n = len(self.conversation_history)
        ctx = f"  ·  Контекст: {n} сообщ." if n > 0 else ""
        self.mode_lbl.setText(f"Режим: {mode}{ctx}")
        if hasattr(self, "model_label"):
            self.model_label.setText(f"Модель: {self.settings['llm_model']}")

    def open_settings(self):
        if self.is_recording:
            self.set_status("Остановите запись перед изменением настроек")
            return
        dialog = SettingsDialog(self.settings, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.settings = dialog.settings_result
        self.toggle.setChecked(bool(self.settings["system_audio_enabled"]))
        self._llm_checked = False
        self._refresh_mode_label()
        self.set_status("Настройки сохранены")

    # ── Recording ─────────────────────────────────────────────────────────────

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.stream is not None or self.system_stream is not None:
            try:
                self._close_audio_streams()
            except Exception:
                self.set_error("предыдущий аудиопоток ещё завершается")
                return
        self._recording_id = getattr(self, "_recording_id", 0) + 1
        self._recording_stop = threading.Event()
        self._mic_signal_at = 0.0
        self._sys_signal_at = 0.0
        with self.frames_lock:
            self.frames = []
            self.full_frames = []
        with self.system_frames_lock:
            self.system_frames = []
            self.full_system_frames = []
        self.settings["system_audio_enabled"] = self.toggle.isChecked()
        self.system_audio_active = False
        self.system_audio_seen = False
        self.system_audio_has_signal = False
        self._system_status_reported = False

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                device=self.settings["input_device"],
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            self.set_error(f"не удалось начать запись: {exc}")
            return

        self.is_recording = True
        self.toggle.setEnabled(False)
        self.recording_started_at = time.time()
        self._last_live_text = ""

        system_warning = None
        if self.settings.get("system_audio_enabled", True):
            try:
                self._start_system_audio_loopback()
                self.system_audio_active = True
                self.set_status("Системный звук: обнаружен")
            except Exception as exc:
                self.system_stream = None
                self.system_pa = None
                self.system_audio_active = False
                system_warning = f"Звук системы недоступен, слушаю микрофон: {str(exc)[:60]}"

        self.listen_btn.configure(
            text="Стоп",
            fg_color=QColor(74, 20, 40),
            border_color=RED,
        )
        if system_warning:
            self.set_status(system_warning)
        elif not self.settings.get("system_audio_enabled", True):
            self.set_status("Слушаю...")

        if self.settings.get("live_transcription", False):
            threading.Thread(target=self._live_transcription_worker,
                             args=(self._recording_id, self._recording_stop), daemon=True).start()

    def audio_callback(self, indata, frames, time_info, status):
        if indata.size and np.max(np.abs(indata.astype(np.int32))) > 120:
            self._mic_signal_at = time.monotonic()
        if status:
            _bridge.set_status_sig.emit("Предупреждение аудиопотока")
        self._update_audio_level(indata)
        with self.frames_lock:
            self.frames.append(indata.copy())
            self.full_frames.append(indata.copy())

    def system_audio_callback(self, indata, frames, time_info, status):
        if status:
            _bridge.set_status_sig.emit("Звук системы нестабилен")
        self._update_audio_level(indata)
        with self.system_frames_lock:
            self.system_frames.append(indata.copy())

    def _start_system_audio_loopback(self):
        if pyaudio is None:
            raise RuntimeError("pyaudiowpatch не установлен")
        self.system_pa = pyaudio.PyAudio()
        device_index = get_system_audio_device(self.settings)
        if device_index is None:
            raise RuntimeError("loopback device не найден")
        info = self.system_pa.get_device_info_by_index(device_index)
        if not info.get("isLoopbackDevice"):
            try:
                info = self.system_pa.get_wasapi_loopback_analogue_by_index(device_index)
                device_index = int(info["index"])
            except Exception as exc:
                raise RuntimeError(f"выбрано не loopback-устройство: {device_index}") from exc
        channels = max(1, min(2, int(info.get("maxInputChannels", 1))))
        source_rate = int(info.get("defaultSampleRate", SAMPLE_RATE))
        self.system_stream = self.system_pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=source_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )
        self._system_reader_thread = threading.Thread(
            target=self._system_audio_reader,
            args=(channels, source_rate, self.system_stream, self._recording_stop),
            daemon=True,
        )
        self._system_reader_thread.start()

    def _system_audio_reader(self, channels, source_rate, stream, stop_event):
        while not stop_event.is_set():
            try:
                # Never block in read while the UI is waiting to close WASAPI.
                available = stream.get_read_available()
                if available <= 0:
                    stop_event.wait(0.01)
                    continue
                data = stream.read(min(1024, available), exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                chunk = resample_audio(chunk, source_rate, SAMPLE_RATE)
                chunk = np.asarray(chunk, dtype=np.int16)
                self.system_audio_seen = True
                peak = int(np.max(np.abs(chunk))) if chunk.size else 0
                if peak > 120:
                    self._sys_signal_at = time.monotonic()
                    self.system_audio_has_signal = True
                    if not self._system_status_reported:
                        self._system_status_reported = True
                        _bridge.set_status_sig.emit("Системный звук: запись идёт")
                self._update_audio_level(chunk)
                with self.system_frames_lock:
                    self.system_frames.append(chunk.copy())
                    self.full_system_frames.append(chunk.copy())
            except Exception as exc:
                self.system_audio_active = False
                _bridge.set_status_sig.emit(f"Звук системы недоступен: {str(exc)[:70]}")
                break

    def _close_audio_streams(self):
        stop_event = getattr(self, "_recording_stop", None)
        if stop_event is not None:
            stop_event.set()
        reader = getattr(self, "_system_reader_thread", None)
        if reader is not None:
            reader.join(timeout=3)
            if reader.is_alive():
                raise RuntimeError("системный аудиопоток ещё завершается; повторите остановку")
            self._system_reader_thread = None
        # No PortAudio read may remain in flight before close/terminate.
        errors = []
        for attr, stop_method in (("stream", "stop"), ("system_stream", "stop_stream")):
            stream = getattr(self, attr, None)
            if stream is not None:
                try:
                    getattr(stream, stop_method)()
                    stream.close()
                    setattr(self, attr, None)
                except Exception as exc:
                    errors.append(type(exc).__name__)
        if self.system_pa is not None and self.system_stream is None:
            self.system_pa.terminate()
            self.system_pa = None
        self.system_audio_active = False
        if errors:
            raise RuntimeError("не удалось освободить аудиопоток: " + ", ".join(errors))

    def _refresh_source_indicators(self):
        now = time.monotonic()
        for label, name, active, last_signal in (
            (self.mic_indicator, "MIC", self.is_recording and self.stream is not None,
             getattr(self, "_mic_signal_at", 0.0)),
            (self.sys_indicator, "SYS", self.is_recording and self.system_audio_active,
             getattr(self, "_sys_signal_at", 0.0)),
        ):
            state = "Выключен" if not active else (
                "Есть сигнал" if last_signal > 0 and now - last_signal < 1.0 else "Поток открыт"
            )
            label.setText(f"{name} · {state}")
            color = GREEN if active and state == "Есть сигнал" else CYAN if active else MUTED
            label.setStyleSheet(f"color: {color.name()}; background: transparent;")

    def _update_audio_level(self, indata):
        try:
            samples = indata.astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)) / 32768.0)
        except Exception:
            return
        level = min(1.0, rms * 9.0)
        self.audio_level = max(level, self.audio_level * 0.82)

    def stop_recording(self):
        self.is_recording = False
        self.toggle.setEnabled(True)
        was_system_audio_active = self.system_audio_active
        self.listen_btn.configure(
            text="Запись",
            fg_color=CYAN_DIM,
            border_color=CYAN,
        )
        self.audio_level = 0.0

        try:
            self._close_audio_streams()
        except Exception as exc:
            self.set_error(f"не удалось остановить запись: {exc}")
            return

        with self.frames_lock:
            frames = list(self.frames)
            self.frames = []
            full_frames = list(self.full_frames)
            self.full_frames = []
        with self.system_frames_lock:
            system_frames = list(self.system_frames)
            self.system_frames = []
            full_system_frames = list(self.full_system_frames)
            self.full_system_frames = []

        if self.settings.get("system_audio_enabled", True) and was_system_audio_active:
            if not system_frames or not self.system_audio_has_signal:
                self.set_status("Системный звук: тишина/нет данных")

        # ── Live mode: after Stop, replace the preview with full final transcription ──
        if self.settings.get("live_transcription", False):
            self.set_status("Запись остановлена")
            if full_frames or full_system_frames:
                audio = self._mix_frames(full_frames, full_system_frames)
                if audio is not None:
                    threading.Thread(target=self._live_final_worker,
                                     args=(audio, self._recording_id), daemon=True).start()
            return

        # ── Normal mode: transcribe everything at once ─────────────────────────────
        if not frames and not system_frames:
            self.set_status("Ничего не распознано")
            return

        audio = self._mix_frames(frames, system_frames)
        if audio is None:
            self.set_status("Ничего не распознано")
            return
        self.set_status("Распознаю...")
        threading.Thread(target=self._transcribe_worker, args=(audio, self._recording_id), daemon=True).start()

    def _mix_frames(self, frames, system_frames):
        """Mix mic and system frame lists into a single int16 array, or None if empty."""
        tracks = []
        system_enabled = self.settings.get("system_audio_enabled", False)
        mix_sources = self.settings.get("mix_mic_and_system_audio", True)
        if frames and (not system_enabled or mix_sources):
            tracks.append(np.concatenate(frames, axis=0))
        if system_frames and system_enabled:
            tracks.append(np.concatenate(system_frames, axis=0))
        return mix_audio_tracks(tracks) if tracks else None

    # ── Transcription ─────────────────────────────────────────────────────────

    def _live_transcription_worker(self, recording_id, stop_event):
        """Background loop: grab a chunk every N seconds and send to Whisper."""
        chunk_sec = max(1, self.settings.get("live_chunk_seconds", 2))
        while not stop_event.wait(chunk_sec):
            if stop_event.is_set():
                break
            with self.frames_lock:
                frames = list(self.frames)
                self.frames = []
            with self.system_frames_lock:
                system_frames = list(self.system_frames)
                self.system_frames = []

            audio = self._mix_frames(frames, system_frames)
            if audio is None:
                continue

            _bridge.set_status_sig.emit("Распознавание в реальном времени...")
            try:
                raw_response, text = self.transcribe_audio_with_raw(audio)
            except Exception as exc:
                log_whisper_diagnostic(
                    self.settings.get("diagnostic_mode", False),
                    "whisper_live_error",
                    error_type=type(exc).__name__,
                )
                if not stop_event.is_set():
                    _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
                continue

            if stop_event.is_set() or recording_id != self._recording_id:
                return
            cleaned = _clean_live_transcription_text(text)
            merged = merge_transcription_chunks(self._last_live_text, cleaned)
            log_whisper_diagnostic(
                self.settings.get("diagnostic_mode", False),
                "whisper_live_chunk",
                raw_bytes=len(raw_response.encode("utf-8")),
                extracted_chars=len(text),
                cleaned_chars=len(cleaned),
                merged_chars=len(merged),
            )
            if merged == self._last_live_text:
                if self.is_recording:
                    _bridge.set_status_sig.emit("Слушаю...")
                continue

            self._last_live_text = merged
            _bridge.append_question_sig.emit(merged)
            _bridge.set_status_sig.emit("Промежуточный текст")

    def _live_final_worker(self, audio, recording_id):
        """Transcribe the full recording after Stop and replace the live preview."""
        _bridge.set_status_sig.emit("Распознаю...")
        try:
            text = self.transcribe_audio(audio)
        except Exception as exc:
            log_whisper_diagnostic(
                self.settings.get("diagnostic_mode", False),
                "whisper_final_error",
                error_type=type(exc).__name__,
            )
            if recording_id == self._recording_id:
                _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
            return
        if recording_id != self._recording_id:
            return
        if not text:
            _bridge.set_status_sig.emit("Ничего не распознано")
            return
        self._last_live_text = text
        _bridge.set_question_sig.emit(text)
        _bridge.set_status_sig.emit("Финальный текст готов")

    def _transcribe_worker(self, audio, recording_id):
        try:
            text = self.transcribe_audio(audio)
        except Exception as exc:
            log_whisper_diagnostic(
                self.settings.get("diagnostic_mode", False),
                "whisper_error",
                error_type=type(exc).__name__,
            )
            if recording_id == self._recording_id:
                _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
            return
        if recording_id != self._recording_id:
            return
        if not text:
            _bridge.set_status_sig.emit("Ничего не распознано")
            return
        _bridge.set_question_sig.emit(text)
        _bridge.set_status_sig.emit("Готов")

    def transcribe_audio(self, audio):
        _, text = self.transcribe_audio_with_raw(audio)
        return text

    def transcribe_audio_with_raw(self, audio):
        wav_buffer = audio_to_wav_bytes(audio, SAMPLE_RATE)
        files = {"file": ("question.wav", wav_buffer, "audio/wav")}
        data = {"model": "whisper-1", "language": self.settings["language"]}
        try:
            response = requests.post(
                self.settings["whisper_url"], files=files, data=data, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(f"Whisper HTTP {code}: {body[:300]}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"ошибка подключения к Whisper: {exc}") from exc
        raw_response = response.text
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Whisper вернул не JSON: {response.text[:300]}") from exc
        return raw_response, payload.get("text", "").strip()

    def _set_question_text(self, text: str):
        self.question_box.setPlainText(text)
        try:
            save_history_entry(self.settings, "transcript.txt", text)
        except OSError:
            self.set_error("не удалось сохранить локальную историю")

    def _set_answer_text(self, text: str):
        self.answer_box.setPlainText(text)

    def copy_to_clipboard(self, text: str):
        if not text.strip():
            self.set_status("Нечего копировать")
            return
        QApplication.clipboard().setText(text)
        self.set_status("Скопировано")

    def copy_question(self):
        self.copy_to_clipboard(self.question_box.toPlainText())

    def copy_answer(self):
        self.copy_to_clipboard(self.answer_box.toPlainText())

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _models_url(self):
        url = self.settings["llm_url"]
        if "/api/chat/completions" in url:
            return url.split("/api/chat/completions")[0] + "/api/models"
        if "/v1/chat/completions" in url:
            return url.split("/v1/chat/completions")[0] + "/v1/models"
        return url.rstrip("/") + "/api/models"

    def _auth_headers(self):
        headers = {}
        if self.settings["openai_api_key"]:
            headers["Authorization"] = f"Bearer {self.settings['openai_api_key']}"
        return headers

    def _check_llm_ready(self):
        response = requests.get(
            self._models_url(),
            headers=self._auth_headers(),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        models = data.get("data", [])
        model_ids = {item.get("id") or item.get("name") for item in models}
        if self.settings["llm_model"] not in model_ids:
            raise RuntimeError(f"модель {self.settings['llm_model']} не найдена на сервере")
        self._llm_checked = True
        return models

    def run_diagnostics(self):
        self.set_status("Проверяю подключение...")
        threading.Thread(target=self._diagnostics_worker, daemon=True).start()

    def _diagnostics_worker(self):
        lines = []
        try:
            lines.append(f"LLM URL: {self.settings['llm_url']}")
            lines.append(f"LLM MODEL: {self.settings['llm_model']}")
            models = self._check_llm_ready()
            lines.append(f"LLM models: OK, найдено {len(models)}")

            payload = build_llm_request(self.settings, "Ответь одним словом: тест")
            response = requests.post(
                self.settings["llm_url"],
                json=payload,
                headers=self._auth_headers(),
                timeout=min(90, self.settings["llm_timeout"]),
            )
            response.raise_for_status()
            actual_model = response.json().get("model", "")
            lines.append(f"LLM chat: OK, model={actual_model}")

            whisper_response = requests.get(self.settings["whisper_url"], timeout=10)
            lines.append(
                f"Whisper endpoint: HTTP {whisper_response.status_code} "
                "(405 для GET допустим)"
            )

            sd.query_devices(self.settings["input_device"], "input")
            lines.append("Микрофон: OK")

            if pyaudio is None:
                lines.append("System audio: pyaudiowpatch не установлен")
            else:
                pa = pyaudio.PyAudio()
                try:
                    device_index = get_system_audio_device(self.settings)
                    info = pa.get_device_info_by_index(device_index)
                    if not info.get("isLoopbackDevice"):
                        info = pa.get_wasapi_loopback_analogue_by_index(device_index)
                    lines.append(f"System audio: OK, {info.get('name')}")
                finally:
                    pa.terminate()
                try:
                    sample = record_system_audio_loopback(self.settings, 1.0, SAMPLE_RATE)
                    peak = int(np.max(np.abs(sample))) if sample is not None and sample.size else 0
                    rms = (
                        float(np.sqrt(np.mean(sample.astype(np.float32) ** 2)))
                        if sample is not None and sample.size else 0.0
                    )
                    if peak > 120:
                        lines.append(f"System audio capture: запись идёт, peak={peak}, rms={rms:.1f}")
                    else:
                        lines.append(f"System audio capture: тишина/нет данных, peak={peak}, rms={rms:.1f}")
                except Exception as exc:
                    lines.append(f"System audio capture: ошибка захвата: {exc}")

            _bridge.set_answer_sig.emit("\n".join(lines))
            _bridge.set_status_sig.emit("Диагностика завершена")
        except Exception as exc:
            lines.append(f"Ошибка: {exc}")
            _bridge.set_answer_sig.emit("\n".join(lines))
            _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")

    def request_answer(self):
        question = self.question_box.toPlainText().strip()
        if not question:
            self.set_status("Ничего не распознано")
            return
        self.set_status("Запрашиваю ответ...")
        threading.Thread(target=self._answer_worker, args=(question,), daemon=True).start()

    def _answer_worker(self, question: str):
        try:
            answer = self.ask_llm(question)
        except Exception as exc:
            _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
            return
        self.conversation_history.add_exchange(question, answer)
        save_history_entry(self.settings, "suggestions.txt", answer)
        _bridge.set_answer_sig.emit(answer)
        _bridge.set_status_sig.emit("Ответ получен")
        QTimer.singleShot(0, self._refresh_mode_label)

    def ask_llm(self, question: str):
        if not self._llm_checked:
            self._check_llm_ready()
        payload = build_llm_request(self.settings, question,
                                    self.conversation_history.get_messages())
        if payload is None:
            raise RuntimeError("некорректный LLM_API_TYPE")
        try:
            response = requests.post(
                self.settings["llm_url"],
                json=payload,
                headers=self._auth_headers(),
                timeout=min(90, self.settings["llm_timeout"]),
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "unknown"
            if code == 404:
                detail = "Проверьте URL и модель. В Open WebUI очистите chat_id."
            elif code in (401, 403):
                detail = "Проверьте API-ключ и права доступа к модели."
            else:
                detail = "Сервер не выполнил запрос. Проверьте настройки LLM и доступность сервиса."
            raise RuntimeError(f"LLM HTTP {code}. {detail}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"ошибка подключения к LLM: {exc}") from exc
        try:
            return parse_llm_response(self.settings, response)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"не удалось разобрать ответ LLM: {response.text[:300]}") from exc

    # ── Misc ─────────────────────────────────────────────────────────────────

    def clear_fields(self):
        self.question_box.clear()
        self.answer_box.clear()
        self.conversation_history.clear()
        self._refresh_mode_label()
        self.set_status("Готов")

    def on_close(self):
        self.is_recording = False
        self._recording_id = getattr(self, "_recording_id", 0) + 1
        try:
            self._close_audio_streams()
        except Exception:
            pass


def main():
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    app.setFont(QFont(system_font_family(), 10))
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    app.setStyle("Fusion")
    win = ModernMeetingHelperGui()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
