import math
import random

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


C = {
    "bg": "#090d17",
    "mid": "#101722",
    "panel": "#121b27",
    "card": "#182332",
    "border": "#53697a",
    "soft_border": "#2a3948",
    "blue": "#49baff",
    "cyan": "#24d8ff",
    "green": "#67d18d",
    "amber": "#ffab43",
    "red": "#ff684f",
    "txt": "#e9f4ff",
    "dim": "#92a5b5",
    "muted": "#64798a",
}


STYLE = f"""
QWidget,QMainWindow{{
    background:{C['bg']};
    color:{C['txt']};
    font-family:"Segoe UI";
    font-size:13px;
}}
QMenuBar{{
    background:#0b0f19;
    color:{C['txt']};
    font-family:"Segoe UI";
    font-size:12px;
    padding:2px 8px;
}}
QMenuBar::item{{padding:4px 8px;}}
QMenuBar::item:selected,QMenu::item:selected{{background:{C['card']};}}
QMenu{{background:{C['mid']};color:{C['txt']};border:1px solid {C['soft_border']};}}
QPushButton,QComboBox,QSpinBox{{
    background:{C['card']};
    border:1px solid {C['soft_border']};
    border-radius:5px;
    color:{C['txt']};
    font-family:"Segoe UI";
    font-size:12px;
    padding:5px 8px;
}}
QPushButton:hover,QComboBox:hover,QSpinBox:hover{{border-color:{C['blue']};}}
QTableWidget{{
    background:#111822;
    border:none;
    color:#d8e7f2;
    selection-background-color:#244766;
    selection-color:#ffffff;
    gridline-color:#273746;
    font-family:"Segoe UI";
    font-size:12px;
}}
QTableWidget::item{{color:#d8e7f2;padding:3px;}}
QHeaderView::section{{
    background:#1b2430;
    color:{C['txt']};
    border:none;
    border-right:1px solid #2c3a48;
    padding:4px;
    font-family:"Segoe UI Semibold";
    font-size:12px;
}}
QSlider::groove:horizontal{{height:4px;background:#2e3945;border-radius:2px;}}
QSlider::sub-page:horizontal{{height:4px;background:{C['blue']};border-radius:2px;}}
QSlider::handle:horizontal{{
    width:14px;
    height:14px;
    background:#ffffff;
    margin:-5px 0;
    border-radius:7px;
}}
QStatusBar{{background:#0b0f19;color:{C['dim']};}}
QLabel{{color:{C['txt']};background:transparent;border:none;}}
QTabWidget::pane{{border:1px solid {C['soft_border']};background:{C['panel']};}}
QTabBar::tab{{
    background:#111822;
    color:{C['txt']};
    padding:7px 12px;
    border:1px solid {C['soft_border']};
}}
QTabBar::tab:selected{{background:#1c2a3a;color:#ffffff;}}
QComboBox QAbstractItemView{{background:#111822;color:{C['txt']};selection-background-color:#244766;}}
"""


def qc(value):
    return QColor(value)


def font(size, bold=False, family="Segoe UI"):
    weight = QFont.Bold if bold else QFont.Normal
    item = QFont(family, size, weight)
    item.setStyleStrategy(QFont.PreferAntialias)
    return item


def mono_font(size, bold=False):
    return font(size, bold, "Cascadia Mono")


def pen(color, width=1):
    line_pen = QPen(qc(color) if isinstance(color, str) else color, width)
    line_pen.setCapStyle(Qt.RoundCap)
    return line_pen


def label(text, size=9, color=C["txt"], bold=False):
    item = QLabel(text)
    item.setFont(font(size, bold))
    item.setStyleSheet(f"QLabel{{color:{color};background:transparent;border:none;}}")
    return item


def apply_palette(app):
    palette = app.palette()
    palette.setColor(QPalette.Window, qc(C["bg"]))
    palette.setColor(QPalette.WindowText, qc(C["txt"]))
    app.setPalette(palette)


def card_style(object_name="panelCard", radius=6):
    return f"""
    QFrame#{object_name}{{
        background:rgba(18,27,39,225);
        border:1px solid {C['soft_border']};
        border-radius:{radius}px;
    }}
    """


class Globe(QWidget):
    """Procedural weather globe used as a lightweight stand-in for Cartopy."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 430)
        self.rotation = 0.0
        self.stream_offset = 0.0
        self.texture = self._build_texture(900)
        self.streams = self._make_streams(90)

        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(32)

    def _tick(self):
        self.rotation = (self.rotation + 0.22) % 360
        self.stream_offset += 0.025
        self.update()

    def _build_texture(self, size):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        p = QPainter(pixmap)
        p.setRenderHint(QPainter.Antialiasing)
        rng = random.Random(13)

        center = QPointF(size / 2, size / 2)
        radius = size * 0.48

        ocean = QRadialGradient(size * 0.42, size * 0.34, radius * 1.2)
        ocean.setColorAt(0.0, QColor("#0f5f91"))
        ocean.setColorAt(0.42, QColor("#063e67"))
        ocean.setColorAt(1.0, QColor("#031327"))
        p.setBrush(ocean)
        p.setPen(Qt.NoPen)
        p.drawEllipse(center, radius, radius)

        continents = [
            [(0.18, 0.44), (0.24, 0.28), (0.37, 0.19), (0.48, 0.26), (0.43, 0.42), (0.32, 0.53)],
            [(0.36, 0.56), (0.45, 0.63), (0.42, 0.84), (0.31, 0.76), (0.26, 0.62)],
            [(0.54, 0.31), (0.70, 0.24), (0.84, 0.34), (0.78, 0.52), (0.61, 0.52)],
            [(0.57, 0.53), (0.72, 0.56), (0.79, 0.78), (0.65, 0.90), (0.55, 0.72)],
            [(0.76, 0.70), (0.88, 0.72), (0.91, 0.84), (0.78, 0.89)],
            [(0.42, 0.13), (0.52, 0.12), (0.58, 0.18), (0.47, 0.21)],
        ]

        for shape in continents:
            path = QPainterPath()
            for index, (x, y) in enumerate(shape):
                point = QPointF(x * size, y * size)
                path.moveTo(point) if index == 0 else path.lineTo(point)
            path.closeSubpath()
            land = QLinearGradient(0, 0, size, size)
            land.setColorAt(0, QColor("#1cb85d"))
            land.setColorAt(0.5, QColor("#d6b46e"))
            land.setColorAt(1, QColor("#f26522"))
            p.setBrush(land)
            p.setPen(pen(QColor(0, 40, 35, 120), 1))
            p.drawPath(path)

        for _ in range(220):
            x = rng.uniform(size * 0.1, size * 0.9)
            y = rng.uniform(size * 0.08, size * 0.9)
            if (x - center.x()) ** 2 + (y - center.y()) ** 2 < (radius * 0.96) ** 2:
                cloud = QColor(235, 250, 255, rng.randint(35, 95))
                p.setPen(pen(cloud, rng.uniform(0.8, 2.4)))
                p.drawLine(QPointF(x, y), QPointF(x + rng.uniform(12, 45), y + rng.uniform(-5, 5)))

        ice = QRadialGradient(size * 0.5, size * 0.12, radius * 0.38)
        ice.setColorAt(0, QColor(255, 255, 255, 235))
        ice.setColorAt(0.8, QColor(225, 245, 255, 90))
        ice.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(ice)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(size * 0.5, size * 0.14), radius * 0.34, radius * 0.13)

        p.end()
        return pixmap

    def _make_streams(self, count):
        rng = random.Random(23)
        streams = []
        for _ in range(count):
            angle = rng.uniform(0, math.tau)
            distance = rng.uniform(0.1, 0.88)
            length = rng.uniform(0.18, 0.55)
            curves = []
            for step in range(18):
                offset = (step / 17) * length
                theta = angle + offset
                wave = math.sin(step * 0.7 + angle) * 0.035
                curves.append((theta, max(0.03, min(0.92, distance + wave))))
            streams.append(curves)
        return streams

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()

        bg = QLinearGradient(0, 0, width, height)
        bg.setColorAt(0, QColor("#0b1024"))
        bg.setColorAt(0.55, QColor("#123f50"))
        bg.setColorAt(1, QColor("#0a0e17"))
        p.fillRect(self.rect(), bg)

        cx = width * 0.49
        cy = height * 0.49
        radius = min(width * 0.36, height * 0.54)

        glow = QRadialGradient(cx, cy, radius * 1.28)
        glow.setColorAt(0.62, QColor(64, 190, 255, 70))
        glow.setColorAt(0.78, QColor(255, 255, 255, 45))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(glow)
        p.drawEllipse(QPointF(cx, cy), radius * 1.28, radius * 1.28)

        clip = QPainterPath()
        clip.addEllipse(QPointF(cx, cy), radius, radius)
        p.setClipPath(clip)

        p.save()
        p.translate(cx, cy)
        p.rotate(self.rotation)
        target = QRectF(-radius, -radius, radius * 2, radius * 2)
        p.drawPixmap(target, self.texture, QRectF(0, 0, self.texture.width(), self.texture.height()))
        p.restore()

        p.setPen(pen(QColor(28, 225, 242, 150), 1.1))
        for stream in self.streams:
            path = QPainterPath()
            for index, (theta, distance) in enumerate(stream):
                theta += self.stream_offset
                x = cx + math.cos(theta) * distance * radius
                y = cy + math.sin(theta) * distance * radius * 0.55
                path.moveTo(x, y) if index == 0 else path.lineTo(x, y)
            p.drawPath(path)

        p.setClipping(False)
        shade = QRadialGradient(cx - radius * 0.25, cy - radius * 0.35, radius * 1.35)
        shade.setColorAt(0.25, QColor(255, 255, 255, 30))
        shade.setColorAt(0.72, QColor(0, 0, 0, 0))
        shade.setColorAt(1, QColor(0, 0, 0, 130))
        p.setBrush(shade)
        p.setPen(pen(QColor(170, 230, 255, 160), 2))
        p.drawEllipse(QPointF(cx, cy), radius, radius)

        p.setFont(font(13, True))
        p.setPen(QColor("#dcecff"))
        p.drawText(int(width * 0.15), int(height * 0.28), "3D GIS Globe")

        self._draw_hud(p, cx + radius * 0.18, cy - radius * 0.44)

    def _draw_hud(self, p, x, y):
        rect = QRectF(x, y, 220, 78)
        p.setPen(pen(QColor(255, 255, 255, 35), 1))
        p.setBrush(QColor(9, 14, 20, 215))
        p.drawRoundedRect(rect, 5, 5)
        p.setPen(QColor("#eaf6ff"))
        p.setFont(font(9, True))
        p.drawText(int(x + 10), int(y + 18), "mini-HUD")
        p.setFont(font(8))
        lines = [
            "Location: London (51.5074N, 0.1278W)",
            "Altitude: 15m",
            "Temp: 14C",
            "Humidity: 78%",
        ]
        for index, text in enumerate(lines):
            p.drawText(int(x + 10), int(y + 35 + index * 12), text)


class Sidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(70)
        self.setStyleSheet(
            f"""
            QWidget{{background:rgba(18,31,44,210);border-right:1px solid {C['soft_border']};}}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setSpacing(5)
        self.buttons = {}

        items = [
            ("Home", "Dashboard"),
            ("Globe", "GIS Globe"),
            ("Chart", "Time-Series\nAnalysis"),
            ("Model", "Predictive\nModeling"),
            ("AI", "Forecast\nAnalysis"),
            ("Export", "Export"),
        ]

        for index, (icon, title) in enumerate(items):
            button = QPushButton(f"{icon}\n{title}")
            button.setFixedHeight(63)
            button.setStyleSheet(
                f"""
                QPushButton{{
                    background:{'rgba(255,255,255,28)' if index == 1 else 'transparent'};
                    color:{C['cyan'] if index == 1 else '#d5e3ee'};
                    border:none;
                    border-left:3px solid {C['cyan'] if index == 1 else 'transparent'};
                    border-radius:3px;
                    font-size:9px;
                }}
                QPushButton:hover{{background:rgba(255,255,255,22);}}
                """
            )
            self.buttons[title.split("\n")[0].lower()] = button
            layout.addWidget(button)

        layout.addStretch()
        settings = QPushButton("Gear\nSettings")
        settings.setFixedHeight(58)
        settings.setStyleSheet("QPushButton{background:transparent;border:none;color:#d5e3ee;font-size:9px;}")
        self.buttons["settings"] = settings
        layout.addWidget(settings)

    def set_active(self, key):
        for name, button in self.buttons.items():
            active = name == key
            button.setStyleSheet(
                f"""
                QPushButton{{
                    background:{'rgba(255,255,255,28)' if active else 'transparent'};
                    color:{C['cyan'] if active else '#d5e3ee'};
                    border:none;
                    border-left:3px solid {C['cyan'] if active else 'transparent'};
                    border-radius:3px;
                    font-size:9px;
                }}
                QPushButton:hover{{background:rgba(255,255,255,22);}}
                """
            )


class Donut(QWidget):
    def __init__(self, value, title, color):
        super().__init__()
        self.value = value
        self.title = title
        self.color = color
        self.setFixedSize(76, 82)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, radius = self.width() // 2, 34, 29
        p.setPen(pen("#40505c", 6))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 210 * 16, -300 * 16)
        try:
            pct = float(str(self.value).replace("%", "").replace("km", "")) / 100
        except ValueError:
            pct = 0.5
        p.setPen(pen(self.color, 6))
        p.drawArc(cx - radius, cy - radius, radius * 2, radius * 2, 210 * 16, int(-300 * 16 * min(pct, 1)))
        p.setPen(QColor(C["txt"]))
        p.setFont(mono_font(12, True))
        p.drawText(0, 0, self.width(), 62, Qt.AlignCenter, str(self.value))
        p.setFont(font(8))
        p.setPen(QColor(C["dim"]))
        p.drawText(0, 61, self.width(), 18, Qt.AlignCenter, self.title)


class MiniChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        rng = random.Random(3)
        self.temp = [18 - i * 0.08 + math.sin(i / 2.3) * 4 + rng.uniform(-1, 1) for i in range(48)]
        self.dew = [10 + math.sin(i / 3.4) * 4 + rng.uniform(-1.2, 1.2) for i in range(48)]

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 36, 12, 22, 24
        pw, ph = w - left - right, h - top - bottom
        vals = self.temp + self.dew
        mn, mx = min(vals) - 2, max(vals) + 3

        p.setPen(pen(QColor(255, 255, 255, 38)))
        for i in range(4):
            y = top + i * ph / 3
            p.drawLine(left, int(y), w - right, int(y))
        p.setFont(font(8))
        p.setPen(QColor(C["dim"]))
        for x, text in [(left, "0h"), (left + pw * 0.25, "6h"), (left + pw * 0.5, "12h"), (left + pw * 0.75, "18h"), (left + pw - 12, "24h")]:
            p.drawText(int(x), h - 5, text)
        p.drawText(6, top + 5, "30")
        p.drawText(12, top + ph - 2, "0")

        def xy(i, value):
            return left + i * pw / (len(self.temp) - 1), top + ph - ((value - mn) / (mx - mn)) * ph

        for values, color in [(self.temp, QColor(C["red"])), (self.dew, QColor(C["blue"]))]:
            path = QPainterPath()
            for i, value in enumerate(values):
                x, y = xy(i, value)
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            p.setPen(pen(color, 1.8))
            p.drawPath(path)

        p.setFont(font(8))
        p.setPen(QColor(C["red"]))
        p.drawText(int(w * 0.32), 14, "Temp (Red)")
        p.setPen(QColor(C["blue"]))
        p.drawText(int(w * 0.60), 14, "Dew Point")


class Sat(QWidget):
    def __init__(self, seed, large=False):
        super().__init__()
        self.seed = seed
        self.large = large
        self.setMinimumSize(86 if not large else 170, 52 if not large else 112)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rng = random.Random(self.seed)
        g = QLinearGradient(0, 0, w, h)
        if self.seed == 40:
            g.setColorAt(0, QColor("#173b2a"))
            g.setColorAt(1, QColor("#0f251c"))
        else:
            g.setColorAt(0, QColor("#8b8f8f"))
            g.setColorAt(1, QColor("#24282b"))
        p.fillRect(self.rect(), g)
        p.setPen(pen(QColor(235, 240, 240, 75), 1.2))
        for _ in range(42 if self.large else 18):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            p.drawArc(x - rng.randint(12, 44), y - rng.randint(12, 44), rng.randint(28, 80), rng.randint(20, 65), rng.randint(0, 360) * 16, rng.randint(30, 120) * 16)
        p.setPen(pen(QColor(255, 255, 255, 80), 1))
        p.drawRoundedRect(0, 0, w - 1, h - 1, 4, 4)


class SectionCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("sectionCard")
        self.setStyleSheet(card_style("sectionCard"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        title_row = QHBoxLayout()
        title_row.addWidget(label(title, 10, C["txt"], True))
        title_row.addStretch()
        title_row.addWidget(label("...", 12, C["dim"], True))
        layout.addLayout(title_row)
        self.body = layout


class RightPanel(QWidget):
    def __init__(self, summary_panel=None, plot_panel=None, controls_widget=None):
        super().__init__()
        self.setFixedWidth(290)
        self.setStyleSheet(f"background:rgba(18,27,37,210);border-left:1px solid {C['soft_border']};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(label("Live Widgets", 16, C["txt"]))

        breakdown = SectionCard("1) Atmospheric Breakdown")
        row = QHBoxLayout()
        for value, title, color in [("78%", "Humidity", C["blue"]), ("3.2", "UV Index", C["amber"]), ("10km", "Visibility", C["green"])]:
            row.addWidget(Donut(value, title, color))
        breakdown.body.addLayout(row)
        layout.addWidget(breakdown)

        chart = SectionCard("2) Multi-Variate Comparison")
        chart.body.addWidget(MiniChart())
        layout.addWidget(chart)

        sat = SectionCard("3) Satellite Feed")
        grid = QGridLayout()
        grid.addWidget(Sat(10, True), 0, 0, 2, 1)
        grid.addWidget(Sat(30), 0, 1)
        grid.addWidget(Sat(40), 1, 1)
        sat.body.addLayout(grid)
        layout.addWidget(sat)

        if controls_widget is not None:
            controls = SectionCard("Dataset Controls")
            controls.body.addWidget(controls_widget)
            layout.addWidget(controls)

        if summary_panel is not None:
            summary = SectionCard("Dataset Summary")
            summary.body.addWidget(summary_panel)
            layout.addWidget(summary)

        if plot_panel is not None:
            analysis = SectionCard("Variable Analysis")
            analysis.body.addWidget(plot_panel)
            layout.addWidget(analysis, 1)
        else:
            layout.addStretch()


class TelemetryGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(122)
        self.setObjectName("telemetryCard")
        self.setStyleSheet(card_style("telemetryCard"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 8)
        title_row = QHBoxLayout()
        title_row.addWidget(label("Data Grid & Telemetry", 10, C["txt"], True))
        title_row.addStretch()
        title_row.addWidget(label("...", 12, C["dim"], True))
        layout.addLayout(title_row)

        self.table = QTableWidget(4, 5)
        self.table.setHorizontalHeaderLabels(["Station ID", "Timestamp", "Barometric Pressure (hPa)", "Wind Speed (m/s)", "Anomalies"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(True)
        rows = [
            ["00000", "2023-02-31 15:36:37.969", "213.50", "1.8", "OK"],
            ["00000", "2023-02-31 15:36:33.891", "213.50", "1.6", "OK"],
            ["00000", "2023-02-31 15:36:33.576", "213.50", "1.4", "OK"],
            ["00000", "2023-02-31 15:36:37.960", "213.50", "1.2", "OK"],
        ]
        for row, values in enumerate(rows):
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 4:
                    item.setForeground(QColor(C["green"]))
                self.table.setItem(row, col, item)
        layout.addWidget(self.table)

        timer = QTimer(self)
        timer.timeout.connect(self._update_wind)
        timer.start(1000)

    def _update_wind(self):
        row = random.randint(0, 3)
        self.table.item(row, 3).setText(f"{random.uniform(1.0, 2.2):.1f}")


class LayerButtons(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background:rgba(8,12,18,165);border:1px solid #53697a;border-radius:7px;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(6)
        for text, color in [("WND", C["cyan"]), ("CLK", C["blue"]), ("TMP", C["red"]), ("HUM", C["blue"]), ("RAD", "#ff3cff"), ("H2O", C["cyan"])]:
            button = QPushButton(text)
            button.setFixedSize(38, 34)
            button.setStyleSheet(
                f"QPushButton{{background:#253240;border:1px solid #4a6575;border-radius:5px;color:{color};font-size:9px;font-weight:bold;}}"
                f"QPushButton:hover{{border-color:{C['cyan']};}}"
            )
            layout.addWidget(button)


class GlobePanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("globePanel")
        self.setStyleSheet(card_style("globePanel", 7))
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.globe = Globe()
        globe_wrap = QWidget()
        globe_wrap.setMinimumHeight(472)
        globe_layout = QGridLayout(globe_wrap)
        globe_layout.setContentsMargins(0, 0, 0, 0)
        globe_layout.addWidget(self.globe, 0, 0)
        globe_layout.addWidget(LayerButtons(), 0, 0, Qt.AlignHCenter | Qt.AlignBottom)

        root.addWidget(globe_wrap, 1)
        root.addWidget(self._time_slider())

    def _time_slider(self):
        wrapper = QWidget()
        wrapper.setStyleSheet("background:rgba(14,19,27,210);border-top:1px solid #53697a;")
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(14, 8, 14, 6)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 72)
        slider.setValue(32)
        layout.addWidget(slider)
        ticks = QHBoxLayout()
        for text in ["0h", "-12h", "-32h", "-45h", "-72h"]:
            ticks.addWidget(label(text, 8, C["dim"]))
            if text != "-72h":
                ticks.addStretch()
        layout.addLayout(ticks)
        return wrapper


class CenterDashboard(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(GlobePanel(), 1)
        layout.addWidget(TelemetryGrid())
