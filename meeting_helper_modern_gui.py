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
    QRect,
)
from PySide6.QtGui import (
    QColor, QPainter, QPainterPath, QLinearGradient, QRadialGradient,
    QFont, QFontMetrics, QPen, QBrush, QCursor, QClipboard, QTextCursor,
    QIcon,
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QSizePolicy, QGraphicsDropShadowEffect, QSizeGrip,
    QDialog, QFormLayout, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QTabWidget, QDialogButtonBox, QMessageBox,
)

from config_manager import (
    CONFIG_PATH,
    DEFAULT_SETTINGS,
    log_path,
    normalize_settings,
    save_config,
)
from meeting_helper import (
    append_line,
    audio_to_wav_bytes,
    build_llm_request,
    ConversationHistory,
    get_system_audio_device,
    load_settings,
    mix_audio_tracks,
    parse_llm_response,
    record_system_audio_loopback,
    resample_audio,
)

SAMPLE_RATE = 16000
CHANNELS = 1
APP_NAME = "PySufler"
APP_DISPLAY_NAME = "PySufler — Суфлёр совещаний"
APP_ID = "PySufler"
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
LIVE_DEBUG_LOG = log_path("whisper_live_debug.log")


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


def log_live_whisper_debug(raw_response: str, extracted: str, cleaned: str, merged: str):
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LIVE_DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] raw_response={raw_response!r}\n")
            fh.write(f"[{stamp}] extracted_text={extracted!r}\n")
            fh.write(f"[{stamp}] cleaned_text={cleaned!r}\n")
            fh.write(f"[{stamp}] merged_text={merged!r}\n\n")
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

    def __init__(self, text, accent: QColor, dim: QColor, parent=None):
        super().__init__(parent)
        self.setFixedSize(188, 54)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._text   = text
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

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
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

        # Text
        p.setPen(TEXT)
        font = QFont("Segoe UI", 14, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, self._text)
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
        self.setFixedSize(52, 28)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._checked = checked
        self._knob_x  = 26.0 if checked else 4.0

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._anim.start(16)

    def isChecked(self): return self._checked

    def setChecked(self, v):
        if v != self._checked:
            self._checked = v
            self.toggled.emit(v)

    def _tick(self):
        target = 26.0 if self._checked else 4.0
        self._knob_x += (target - self._knob_x) * 0.18
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._checked = not self._checked
            self.toggled.emit(self._checked)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        track_col = CYAN if self._checked else QColor(40, 50, 80)
        track_c = QColor(track_col)
        track_c.setAlpha(200)
        p.setPen(Qt.NoPen)
        p.setBrush(track_c)
        p.drawRoundedRect(QRectF(0, 4, w, h - 8), (h - 8) / 2, (h - 8) / 2)

        p.setBrush(QColor(240, 250, 255))
        p.drawEllipse(QPointF(self._knob_x + 10, h / 2), 10, 10)
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
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки PySufler")
        self.setMinimumSize(720, 560)
        self.settings_result = None
        self._build_ui(normalize_settings(settings))

    def _build_ui(self, settings: dict):
        self.setStyleSheet("""
            QDialog { background: #07101f; color: #e5e7eb; }
            QLabel { color: #cfe4ff; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #0f1b2e; color: #e5e7eb; border: 1px solid #27476f;
                border-radius: 8px; padding: 7px 9px;
            }
            QPushButton {
                background: #13233a; color: #e5e7eb; border: 1px solid #31547d;
                border-radius: 9px; padding: 8px 12px;
            }
            QPushButton:hover { border-color: #38bdf8; background: #17304f; }
            QTabWidget::pane { border: 1px solid #1f3a5b; border-radius: 10px; }
            QTabBar::tab {
                background: #101b2e; color: #94a3b8; padding: 9px 14px;
                border-top-left-radius: 8px; border-top-right-radius: 8px;
            }
            QTabBar::tab:selected { color: #ffffff; background: #18294a; }
            QCheckBox { color: #dbeafe; spacing: 8px; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QLabel(f"Настройки сохраняются в {CONFIG_PATH}")
        title.setWordWrap(True)
        title.setStyleSheet("color: #94a3b8;")
        root.addWidget(title)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        self.llm_api_type = QComboBox()
        self.llm_api_type.addItems(["openwebui", "openai", "ollama", "ollama_generate"])
        self.llm_api_type.setCurrentText(settings["llm_api_type"])
        self.llm_url = QLineEdit(settings["llm_url"])
        self.llm_model = QComboBox()
        self.llm_model.setEditable(True)
        self.llm_model.addItem(settings["llm_model"])
        self.llm_model.setCurrentText(settings["llm_model"])
        self.api_key = QLineEdit(settings["openai_api_key"])
        self.api_key.setEchoMode(QLineEdit.Password)
        self.show_key = QCheckBox("Показать ключ")
        self.show_key.toggled.connect(lambda checked: self.api_key.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password))
        self.chat_id = QLineEdit(settings["openwebui_chat_id"])
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(float(settings.get("temperature", 0.2)))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 8192)
        self.max_tokens.setValue(int(settings.get("max_tokens", 300)))
        self.llm_timeout = QSpinBox()
        self.llm_timeout.setRange(5, 1200)
        self.llm_timeout.setValue(int(settings["llm_timeout"]))

        llm_tab = QWidget()
        llm_form = QFormLayout(llm_tab)
        llm_form.setLabelAlignment(Qt.AlignRight)
        llm_form.addRow("Тип API", self.llm_api_type)
        llm_form.addRow("LLM URL", self.llm_url)
        model_row = QHBoxLayout()
        model_row.addWidget(self.llm_model, 1)
        refresh_btn = QPushButton("Обновить модели")
        refresh_btn.clicked.connect(self.refresh_models)
        model_row.addWidget(refresh_btn)
        llm_form.addRow("Модель", model_row)
        llm_form.addRow("OPENAI_API_KEY", self.api_key)
        llm_form.addRow("", self.show_key)
        llm_form.addRow("OpenWebUI chat_id", self.chat_id)
        llm_form.addRow("Temperature", self.temperature)
        llm_form.addRow("Max tokens", self.max_tokens)
        llm_form.addRow("Timeout, sec", self.llm_timeout)
        tabs.addTab(llm_tab, "LLM")

        self.whisper_url = QLineEdit(settings["whisper_url"])
        self.language = QLineEdit(settings["language"])
        self.chunk_seconds = QSpinBox()
        self.chunk_seconds.setRange(1, 60)
        self.chunk_seconds.setValue(int(settings["chunk_seconds"]))
        self.buffer_minutes = QSpinBox()
        self.buffer_minutes.setRange(1, 240)
        self.buffer_minutes.setValue(int(settings["buffer_minutes"]))
        whisper_tab = QWidget()
        whisper_form = QFormLayout(whisper_tab)
        whisper_form.addRow("Whisper URL", self.whisper_url)
        whisper_form.addRow("Язык", self.language)
        whisper_form.addRow("Chunk seconds", self.chunk_seconds)
        whisper_form.addRow("Buffer minutes", self.buffer_minutes)
        tabs.addTab(whisper_tab, "Whisper")

        self.input_device = QComboBox()
        self.input_device.addItem("Авто", None)
        self.system_audio_device = QComboBox()
        self.system_audio_device.addItem("Авто WASAPI loopback", None)
        self._fill_audio_devices(settings)
        self.system_audio_enabled = QCheckBox("Слушать собеседника")
        self.system_audio_enabled.setChecked(bool(settings["system_audio_enabled"]))
        self.auto_system_loopback = QCheckBox("Автоматически выбирать WASAPI loopback")
        self.auto_system_loopback.setChecked(bool(settings.get("auto_system_loopback", True)))
        self.mix_audio = QCheckBox("Смешивать микрофон + системный звук")
        self.mix_audio.setChecked(bool(settings["mix_mic_and_system_audio"]))
        audio_tab = QWidget()
        audio_form = QFormLayout(audio_tab)
        audio_form.addRow("Микрофон", self.input_device)
        audio_form.addRow("", self.system_audio_enabled)
        audio_form.addRow("", self.auto_system_loopback)
        audio_form.addRow("System audio", self.system_audio_device)
        audio_form.addRow("", self.mix_audio)
        tabs.addTab(audio_tab, "Audio")

        self.live_transcription = QCheckBox("Live-распознавание")
        self.live_transcription.setChecked(bool(settings["live_transcription"]))
        self.live_chunk_seconds = QSpinBox()
        self.live_chunk_seconds.setRange(1, 20)
        self.live_chunk_seconds.setValue(int(settings["live_chunk_seconds"]))
        live_tab = QWidget()
        live_form = QFormLayout(live_tab)
        live_form.addRow("", self.live_transcription)
        live_form.addRow("Live chunk seconds", self.live_chunk_seconds)
        tabs.addTab(live_tab, "Live")

        self.hotkey = QLineEdit(settings["hotkey"])
        self.prompt_mode = QComboBox()
        self.prompt_mode.addItems(["last", "buffer"])
        self.prompt_mode.setCurrentText(settings["prompt_mode"])
        self.last_lines = QSpinBox()
        self.last_lines.setRange(1, 100)
        self.last_lines.setValue(int(settings["last_lines"]))
        advanced_tab = QWidget()
        advanced_form = QFormLayout(advanced_tab)
        advanced_form.addRow("Hotkey", self.hotkey)
        advanced_form.addRow("Prompt mode", self.prompt_mode)
        advanced_form.addRow("Last lines", self.last_lines)
        tabs.addTab(advanced_tab, "Advanced")

        bottom = QHBoxLayout()
        defaults_btn = QPushButton("По умолчанию")
        defaults_btn.clicked.connect(self.restore_defaults)
        check_btn = QPushButton("Проверить подключение")
        check_btn.clicked.connect(self.check_connection)
        bottom.addWidget(defaults_btn)
        bottom.addWidget(check_btn)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        root.addLayout(bottom)

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

    def _collect(self) -> dict:
        return normalize_settings({
            "whisper_url": self.whisper_url.text().strip(),
            "llm_url": self.llm_url.text().strip(),
            "llm_api_type": self.llm_api_type.currentText().strip().lower(),
            "llm_model": self.llm_model.currentText().strip(),
            "openai_api_key": self.api_key.text().strip(),
            "openwebui_chat_id": self.chat_id.text().strip() or "meeting-helper",
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
        })

    def restore_defaults(self):
        self.settings_result = normalize_settings(DEFAULT_SETTINGS)
        save_config(self.settings_result)
        QMessageBox.information(self, "PySufler", "Настройки сброшены к безопасным значениям по умолчанию.")
        self.accept()

    def refresh_models(self):
        try:
            models = self._request_models()
        except Exception as exc:
            QMessageBox.warning(self, "PySufler", f"Не удалось получить модели:\n{exc}")
            return
        current = self.llm_model.currentText()
        self.llm_model.clear()
        for model in models:
            self.llm_model.addItem(model)
        if current:
            self.llm_model.setCurrentText(current)

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
        try:
            models = self._request_models()
            lines.append(f"LLM: OK, моделей найдено: {len(models)}")
        except Exception as exc:
            lines.append(f"LLM: ошибка: {exc}")
        try:
            response = requests.get(self.whisper_url.text().strip(), timeout=5)
            lines.append(f"Whisper endpoint: HTTP {response.status_code}")
        except Exception as exc:
            lines.append(f"Whisper endpoint: {exc}")
        QMessageBox.information(self, "Проверка подключения", "\n".join(lines))

    def accept(self):
        if self.settings_result is None:
            self.settings_result = self._collect()
            save_config(self.settings_result)
        super().accept()


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
        start_w = min(1200, max(1050, available.width() - 80))
        start_h = min(950, max(760, available.height() - 80))
        min_h = 740 if start_h < 840 else 800
        self.resize(start_w, start_h)
        self.setMinimumSize(1050, min_h)

        self.settings = load_settings()
        self.settings["system_audio_enabled"] = env_bool("SYSTEM_AUDIO_ENABLED", True)
        self.settings["mix_mic_and_system_audio"] = env_bool("MIX_MIC_AND_SYSTEM_AUDIO", True)

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
        outer.addWidget(content)

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
        lbl.setFont(QFont("Segoe UI", size, w))
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
        title_col.addWidget(self._label(APP_DISPLAY_NAME, 30, True))
        title_col.addWidget(self._label("локальный помощник PySufler для быстрых ответов", 13,
                                        color=QColor("#8aabcc")))
        self.model_label = self._label(f"Модель: {self.settings['llm_model']}", 10,
                                       color=QColor("#6f8fb1"))
        title_col.addWidget(self.model_label)
        row.addLayout(title_col)
        row.addStretch()

        self.settings_btn = NeonButton("⚙", QColor(70, 110, 155), QColor(10, 18, 36))
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

        self.listen_btn = NeonButton("◉  Запись", CYAN, CYAN_DIM)
        self.listen_btn.set_size(190, 54)
        self.listen_btn.clicked.connect(self.toggle_recording)
        row.addWidget(self.listen_btn)

        self.answer_btn = NeonButton("⚡  Ответ", PURPLE, PURPLE_DIM)
        self.answer_btn.set_size(170, 54)
        self.answer_btn.clicked.connect(self.request_answer)
        row.addWidget(self.answer_btn)

        self.clear_btn = NeonButton("◌  Очистить", QColor(58, 80, 112), QColor(16, 28, 50))
        self.clear_btn.set_size(155, 54)
        self.clear_btn.clicked.connect(self.clear_fields)
        row.addWidget(self.clear_btn)

        self.check_btn = NeonButton("✓  Проверить", QColor(68, 96, 130), QColor(14, 26, 46))
        self.check_btn.set_size(145, 54)
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
        row.addLayout(toggle_col)

        vbox.addLayout(row)

    def _build_config_bar(self, vbox):
        label = self._label(f"Модель: {self.settings['llm_model']}", 11,
                            color=QColor("#7fa7c8"))
        label.setWordWrap(True)
        vbox.addWidget(label)

    def _build_question_card(self, vbox):
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
        self.status_lbl.setText(text)
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
        mode = ("микрофон + звук системы"
                if self.settings["system_audio_enabled"] else "микрофон")
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
        with self.frames_lock:
            self.frames = []
            self.full_frames = []
        with self.system_frames_lock:
            self.system_frames = []
            self.full_system_frames = []
        self.settings["system_audio_enabled"] = self.toggle.isChecked()
        self.settings["mix_mic_and_system_audio"] = env_bool("MIX_MIC_AND_SYSTEM_AUDIO", True)
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
            text="■  Стоп",
            fg_color=QColor(74, 20, 40),
            border_color=RED,
        )
        if system_warning:
            self.set_status(system_warning)
        elif not self.settings.get("system_audio_enabled", True):
            self.set_status("Слушаю...")

        if self.settings.get("live_transcription", False):
            threading.Thread(target=self._live_transcription_worker,
                             daemon=True).start()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            QTimer.singleShot(0, lambda: self.set_status(f"Предупреждение аудио: {status}"))
        self._update_audio_level(indata)
        with self.frames_lock:
            self.frames.append(indata.copy())
            self.full_frames.append(indata.copy())

    def system_audio_callback(self, indata, frames, time_info, status):
        if status:
            QTimer.singleShot(0, lambda: self.set_status("Звук системы нестабилен"))
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
        threading.Thread(
            target=self._system_audio_reader,
            args=(channels, source_rate),
            daemon=True,
        ).start()

    def _system_audio_reader(self, channels, source_rate):
        while self.is_recording and self.system_stream is not None:
            try:
                data = self.system_stream.read(1024, exception_on_overflow=False)
                chunk = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                chunk = resample_audio(chunk, source_rate, SAMPLE_RATE)
                chunk = np.asarray(chunk, dtype=np.int16)
                self.system_audio_seen = True
                peak = int(np.max(np.abs(chunk))) if chunk.size else 0
                if peak > 120:
                    self.system_audio_has_signal = True
                    if not self._system_status_reported:
                        self._system_status_reported = True
                        _bridge.set_status_sig.emit("Системный звук: запись идёт")
                self._update_audio_level(chunk)
                with self.system_frames_lock:
                    self.system_frames.append(chunk.copy())
                    self.full_system_frames.append(chunk.copy())
            except Exception as exc:
                _bridge.set_status_sig.emit(f"Звук системы недоступен: {str(exc)[:70]}")
                break

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
        was_system_audio_active = self.system_audio_active
        self.listen_btn.configure(
            text="◉  Запись",
            fg_color=CYAN_DIM,
            border_color=CYAN,
        )
        self.audio_level = 0.0

        try:
            if self.stream is not None:
                self.stream.stop(); self.stream.close()
            if self.system_stream is not None:
                self.system_stream.stop_stream(); self.system_stream.close()
        except Exception as exc:
            self.set_error(f"не удалось остановить запись: {exc}")
            return
        finally:
            self.stream = None
            self.system_stream = None
            if self.system_pa is not None:
                try:
                    self.system_pa.terminate()
                except Exception:
                    pass
            self.system_pa = None
            self.system_audio_active = False

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
                                     args=(audio,), daemon=True).start()
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
        threading.Thread(target=self._transcribe_worker, args=(audio,), daemon=True).start()

    def _mix_frames(self, frames, system_frames):
        """Mix mic and system frame lists into a single int16 array, or None if empty."""
        tracks = []
        if frames:
            tracks.append(np.concatenate(frames, axis=0))
        if system_frames and (
            self.settings.get("mix_mic_and_system_audio", True) or not frames
        ):
            tracks.append(np.concatenate(system_frames, axis=0))
        return mix_audio_tracks(tracks) if tracks else None

    # ── Transcription ─────────────────────────────────────────────────────────

    def _live_transcription_worker(self):
        """Background loop: grab a chunk every N seconds and send to Whisper."""
        chunk_sec = max(1, self.settings.get("live_chunk_seconds", 2))
        while self.is_recording:
            time.sleep(chunk_sec)
            if not self.is_recording:
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
                _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
                continue

            cleaned = _clean_live_transcription_text(text)
            merged = merge_transcription_chunks(self._last_live_text, cleaned)
            log_live_whisper_debug(raw_response, text, cleaned, merged)
            if merged == self._last_live_text:
                if self.is_recording:
                    _bridge.set_status_sig.emit("Слушаю...")
                continue

            self._last_live_text = merged
            _bridge.append_question_sig.emit(merged)
            _bridge.set_status_sig.emit("Промежуточный текст")

    def _live_final_worker(self, audio):
        """Transcribe the full recording after Stop and replace the live preview."""
        _bridge.set_status_sig.emit("Распознаю...")
        try:
            text = self.transcribe_audio(audio)
        except Exception as exc:
            _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
            return
        if not text:
            _bridge.set_status_sig.emit("Ничего не распознано")
            return
        self._last_live_text = text
        _bridge.set_question_sig.emit(text)
        _bridge.set_status_sig.emit("Финальный текст готов")

    def _transcribe_worker(self, audio):
        try:
            text = self.transcribe_audio(audio)
        except Exception as exc:
            _bridge.set_status_sig.emit(f"Ошибка: {str(exc)[:90]}")
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
        append_line("transcript.txt", text)

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
        append_line("suggestions.txt", answer)
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
            body = exc.response.text if exc.response is not None else ""
            raise RuntimeError(f"LLM HTTP {code}: {body[:300]}") from exc
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
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
        if self.system_stream is not None:
            try:
                self.system_stream.stop_stream(); self.system_stream.close()
            except Exception:
                pass
        if self.system_pa is not None:
            try:
                self.system_pa.terminate()
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
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    app.setStyle("Fusion")
    win = ModernMeetingHelperGui()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
