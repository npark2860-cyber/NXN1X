// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 SnapRYU contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "performance_overlay.h"
#include "process_utilization.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <QApplication>
#include <QCloseEvent>
#include <QFontDatabase>
#include <QFontMetrics>
#include <QFontMetricsF>
#include <QHideEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QShowEvent>
#include <QWheelEvent>
#include "common/settings.h"
#include "common/dlss_fg_setting.h"
#include "core/nvn/nvn_layer.h"
#include "core/perf_stats.h"
#include "main_window.h"
#include "qt_common/config/uisettings.h"
#include "video_core/shader_notify.h"
#include "video_core/texture_cache/res_info.h"   // resoverlay
#include "video_core/renderer_vulkan/fg_status.h"   // fgtoggle

// resoverlay: the NVN host's last presented frame: largest render target width, height, render scale (gfx_vk_tex.c; weak: builds without the host)
extern "C" __attribute__((weak)) void nvnh_res_info(unsigned out[3]);
extern "C" __attribute__((weak)) void nvnh_dlss_info(unsigned out[6]);   // dlsstoggle (gfx_vk_exec.c): on, running, render w h, DLSS input w, output h

namespace {
constexpr int Padding = 10;
constexpr int Margin = 8;
constexpr int RowHeight = 18;
constexpr int Radius = 8;
constexpr int UnitGap = 6;
constexpr int LabelGap = 8;
constexpr int OpacityStep = 15;
constexpr int MinOpacity = 40;
constexpr int MaxOpacity = 255;
constexpr int SpinnerSize = 14;
constexpr int SpinnerSpokes = 12;
constexpr int SpinnerStepMs = 80; // one turn per ~1 s
constexpr int SpinnerAlphaStep = 19; // head 255, last trailing spoke 46
const QColor Accent{240, 128, 60};
const QColor Dim{185, 185, 185};
const QColor Divider{255, 255, 255, 60};

bool Valid(double value) {
    return std::isfinite(value) && value >= 0.0 && value < 10000.0;
}
QString Number(double value, int decimals) {
    return Valid(value) ? QString::number(value, 'f', decimals) : QStringLiteral("--");
}
QString Percent(double value, int decimals) {
    return std::isfinite(value) && value >= 0.0
               ? QStringLiteral("%1%").arg(value, 0, 'f', decimals) : QStringLiteral("--");
}
QString Megabytes(std::int64_t bytes) {
    return bytes >= 0 ? QStringLiteral("%1 MB").arg(bytes / 1'000'000) : QStringLiteral("--");
}
// Numbers in fixed columns (same rule as the native host overlay, r91): each digit's ink sits
// centred in a cell as wide as the widest digit's ink plus a 1-2 px gap, and other characters
// keep their own width, so a changing value keeps its place without the digits spreading apart
qreal DigitGap(const QFont& font) {
    return font.pixelSize() >= 22 ? 2.0 : 1.0;
}
// The glyph outline's exact extent (font metrics round ink boxes to whole pixels)
QRectF Ink(const QFont& font, QChar c) {
    QPainterPath path;
    path.addText(0.0, 0.0, font, QString(c));
    return path.boundingRect();
}
qreal DigitCell(const QFont& font) {
    qreal ink = 0.0;
    for (char d = '0'; d <= '9'; ++d) {
        ink = std::max(ink, Ink(font, QLatin1Char(d)).width());
    }
    return ink + DigitGap(font);
}
// Left edge for drawing a digit so its ink is centred in the cell starting at x
qreal DigitX(const QFont& font, QChar c, qreal x, qreal cell) {
    const QRectF ink = Ink(font, c);
    return x + (cell - ink.width()) / 2.0 - ink.left();
}
qreal TabWidthF(const QFont& font, const QString& text) {
    const QFontMetricsF metrics{font};
    const qreal cell = DigitCell(font);
    qreal width = 0.0;
    for (const QChar c : text) {
        width += c.isDigit() ? cell : metrics.horizontalAdvance(c);
    }
    return width;
}
int TabWidth(const QFont& font, const QString& text) {
    return static_cast<int>(std::ceil(TabWidthF(font, text)));
}
// Draws text starting at a baseline point
void DrawTab(QPainter& painter, const QFont& font, QPointF origin, const QString& text) {
    const QFontMetricsF metrics{font};
    const qreal cell = DigitCell(font);
    qreal x = origin.x();
    for (const QChar c : text) {
        const bool digit = c.isDigit();
        painter.drawText(QPointF{digit ? DigitX(font, c, x, cell) : x, origin.y()}, QString(c));
        x += digit ? cell : metrics.horizontalAdvance(c);
    }
}
// Draws text in a row, vertically centred like drawText(rect, AlignLeft | AlignVCenter)
void DrawTabRow(QPainter& painter, const QFont& font, qreal x, int top, const QString& text) {
    const QFontMetricsF metrics{font};
    const qreal cell = DigitCell(font);
    for (const QChar c : text) {
        const bool digit = c.isDigit();
        const qreal left = digit ? DigitX(font, c, x, cell) : x;
        painter.drawText(QRectF{left, static_cast<qreal>(top), cell, static_cast<qreal>(RowHeight)},
                         Qt::AlignLeft | Qt::AlignVCenter | Qt::TextDontClip, QString(c));
        x += digit ? cell : metrics.horizontalAdvance(c);
    }
}

QString Share(std::int64_t part, std::int64_t whole) {
    if (part < 0 || whole <= 0) return {};
    return QStringLiteral("(%1%)").arg(std::lround(100.0 * static_cast<double>(part) /
                                                   static_cast<double>(whole)));
}

// Pretendard Std (SIL OFL 1.1) is bundled so recordings look the same on every machine.
QString OverlayFamily() {
    static const QString family = [] {
        QString name;
        for (const auto weight : {"Regular", "Medium", "SemiBold", "Bold"}) {
            const int id = QFontDatabase::addApplicationFont(
                QStringLiteral(":/fonts/PretendardStd-%1.otf").arg(QLatin1String(weight)));
            if (id < 0 || !name.isEmpty()) continue;
            const auto families = QFontDatabase::applicationFontFamilies(id);
            if (!families.isEmpty()) name = families.front();
        }
        return name.isEmpty() ? QStringLiteral("Segoe UI") : name;
    }();
    return family;
}
QFont OverlayFont(int pixel_size, QFont::Weight weight) {
    QFont font{OverlayFamily()};
    font.setPixelSize(pixel_size);
    font.setWeight(weight);
    return font;
}
} // namespace

PerformanceOverlay::PerformanceOverlay(MainWindow* parent, QWidget* anchor)
    : QWidget(parent), m_mainWindow{parent}, m_anchor{anchor ? anchor : parent},
      m_big_font{OverlayFont(34, QFont::Bold)}, m_compact_font{OverlayFont(40, QFont::Bold)},
      m_unit_font{OverlayFont(12, QFont::DemiBold)}, m_label_font{OverlayFont(12, QFont::Medium)},
      m_value_font{OverlayFont(12, QFont::Normal)}, m_tag_font{OverlayFont(11, QFont::Medium)},
      m_fps_only{UISettings::values.perf_overlay_fps_only.GetValue()} {
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool | Qt::WindowDoesNotAcceptFocus);
    setFont(m_value_font);
    setCursor(Qt::SizeAllCursor);
    setAccessibleName(tr("Performance overlay"));
    setToolTip(tr("FPS: accepted Vulkan presentation requests, before DLSS interpolation; "
                  "not a monitor scanout measurement.\n"
                  "Game: game-rendered FPS before frame generation; shown when DLSS is "
                  "selected for NVN.\n"
                  "CPU: process CPU time divided by total logical CPU capacity.\n"
                  "GPU: this process's busiest Windows GPU engine.\n"
                  "RAM: process resident working set (share of physical memory). VRAM: "
                  "driver-estimated device-local usage (share of the current budget, not "
                  "physical capacity). MB = 1,000,000 bytes.\n"
                  "CPU/GPU/RAM sampled about once per second; -- means unavailable or warming up.\n"
                  "Mode: active CPU backend. Driver: active Vulkan backend and driver.\n"
                  "Shaders: shaders built since the spinner appeared; the spinner turns "
                  "while shaders are being built.\n"
                  "Drag to move. Click to show FPS only. Scroll to change background opacity."));
    m_spin_timer.setInterval(SpinnerStepMs);
    connect(&m_spin_timer, &QTimer::timeout, this, [this] {
        m_spin_step = (m_spin_step + 1) % SpinnerSpokes;
        update();
    });
    if (m_anchor) m_anchor->installEventFilter(this);
    updateLines(std::numeric_limits<double>::quiet_NaN());
    if (parent) {
        connect(parent, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
        // Reuse the existing 500 ms UI sample; never reset core stats a second time.
        connect(parent, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);
    }
}

PerformanceOverlay::~PerformanceOverlay() {
    stopSampling();
}

void PerformanceOverlay::SetDriverText(const QString& text) {
    if (m_driver == text) return;
    m_driver = text;
    updateLines(m_native_fps);
}

bool PerformanceOverlay::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_anchor && (event->type() == QEvent::Resize ||
                               event->type() == QEvent::Move || event->type() == QEvent::Show)) {
        resetPosition();
    }
    return QWidget::eventFilter(watched, event);
}

void PerformanceOverlay::showEvent(QShowEvent* event) {
    m_present_sampler.Reset();
    m_fps = QStringLiteral("--");
    m_fps_whole = QStringLiteral("--");
    m_shaders_last_started = -1;
    m_shader_burst_base = -1;
    m_shader_burst = 0;
    if (!m_sampling) {
        Frontend::ProcessUtilization::Instance().AddClient();
        m_sampling = true;
    }
    updateLines(std::numeric_limits<double>::quiet_NaN());
    QWidget::showEvent(event);
}

void PerformanceOverlay::hideEvent(QHideEvent* event) {
    stopSampling();
    QWidget::hideEvent(event);
}

void PerformanceOverlay::stopSampling() {
    m_present_sampler.Reset();
    m_spin_timer.stop();
    if (m_sampling) {
        Frontend::ProcessUtilization::Instance().RemoveClient();
        m_sampling = false;
    }
}

void PerformanceOverlay::resetPosition(const QPoint&) {
    QWidget* anchor = m_anchor ? m_anchor.data() : m_mainWindow;
    if (!anchor) return;
    const auto origin = anchor->mapToGlobal(QPoint{});
    const QPoint desired{anchor->width() - width() - Margin + m_drag_offset.x(),
                         Margin + m_drag_offset.y()};
    const QPoint bounded{std::clamp(desired.x(), 0, std::max(0, anchor->width() - width())),
                         std::clamp(desired.y(), 0, std::max(0, anchor->height() - height()))};
    move(origin + bounded);
}

void PerformanceOverlay::UpdateRendererStats(const VideoCore::RendererPerformanceStats& stats) {
    if (!m_sampling) return;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto rates = m_present_sampler.Sample(stats.presents, now);
    m_fps = Number(rates.total, 2);
    m_fps_whole = Number(rates.total, 0);
    m_vram_used = stats.memory_available ? static_cast<std::int64_t>(stats.memory_used) : -1;
    m_vram_budget = stats.memory_available ? static_cast<std::int64_t>(stats.memory_budget) : -1;
}

void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                    const VideoCore::ShaderNotify& shaders) {
    const int completed = shaders.ShadersCompleted();
    const int started = shaders.ShadersStarted();
    // First sample, or a new game reset the counters: only shaders still in flight count.
    if (m_shaders_last_started < 0 || started < m_shaders_last_started) {
        m_shaders_last_started = completed;
        m_shader_burst_base = -1;
    }
    // Building = a shader is in flight now, or one started since the last 500 ms sample.
    const bool building = started > completed || started != m_shaders_last_started;
    if (building && m_shader_burst_base < 0) {
        m_shader_burst_base = m_shaders_last_started;
    } else if (!building) {
        m_shader_burst_base = -1;
    }
    m_shaders_last_started = started;
    m_shader_burst = building ? started - m_shader_burst_base : 0;
    if (building && m_sampling) {
        if (!m_spin_timer.isActive()) m_spin_timer.start();
    } else {
        m_spin_timer.stop();
    }
    updateLines(results.average_game_fps);
}

void PerformanceOverlay::updateLines(double native_fps) {
    m_native_fps = native_fps;
    const auto sample = m_sampling ? Frontend::ProcessUtilization::Instance().Read()
                                  : Frontend::ProcessUtilizationSnapshot{};
    // NCE runs the game; NVN = the host NVN layer draws (Core::NvnLayer::Wanted, false again after a fallback), EMULATED = the
    // GPU is emulated as usual (NPARK 11-12)
    const auto mode = !Settings::IsNceEnabled()     ? QStringLiteral("Dynamic")
                      : Core::NvnLayer::Wanted()     ? QStringLiteral("NCE (NVN)")
                                                     : QStringLiteral("NCE (EMULATED)");
    m_show_game = Settings::dlss_frame_generation.GetValue() && Core::NvnLayer::Wanted();
    m_game_fps = QStringLiteral("Game %1").arg(Number(native_fps, 1));
    m_tag.clear();
    if (m_fps_only) {
        m_rows = {{QStringLiteral("GPU"), Percent(sample.gpu_percent, 0), {}}};
    } else {
        m_rows = {{QStringLiteral("CPU"), Percent(sample.cpu_percent, 1), {}},
                  {QStringLiteral("GPU"), Percent(sample.gpu_percent, 1), {}},
                  {QStringLiteral("RAM"), Megabytes(sample.ram_bytes),
                   Share(sample.ram_bytes, Frontend::TotalPhysicalMemoryBytes())},
                  {QStringLiteral("VRAM"), Megabytes(m_vram_used), Share(m_vram_used, m_vram_budget)},
                  {QStringLiteral("Mode"), mode, {}}};
        if (!m_driver.isEmpty()) {
            m_rows.push_back({QStringLiteral("Driver"), m_driver, {}});
        }
        // resoverlay (11-21, NPARK): internal (the largest render target the game drew into last frame) -> render (what the host
        // draws: the NVN path applies only the exact 2x scale) -> output (the window, upscaled by the present filter)
        {   // every game: the NVN host's numbers in NVN mode, the emulated GPU's (texture cache) otherwise
            unsigned in_w = 0, in_h = 0, r_h = 0; QString path;
            const float set = Settings::values.resolution_info.up_factor;
            if (Core::NvnLayer::Wanted()) {
                unsigned r[3] = {0, 0, 1};
                if (nvnh_res_info) nvnh_res_info(r);
                in_w = r[0]; in_h = r[1]; r_h = r[1] * r[2];
                path = r[2] == 2 ? QStringLiteral(" x2") : set != 1.0f ? QStringLiteral(" x1!") : QString{};   // x1!: a scale is set that the NVN path does not apply
            } else {
                in_w = VideoCommon::ResInfo::last_w; in_h = VideoCommon::ResInfo::last_h; r_h = VideoCommon::ResInfo::last_rh;
                if (in_h && r_h != in_h) path = QStringLiteral(" x%1").arg(Number(double(r_h) / in_h, 2));
            }
            if (in_w && in_h) {   // resoverlay2 (NPARK): Res 3D / game output / screen area, heights only so the panel keeps its width
                const qreal dpr = m_anchor ? m_anchor->devicePixelRatioF() : 1.0;
                const double win_w = m_anchor ? m_anchor->width() * dpr : 0.0, win_h = m_anchor ? m_anchor->height() * dpr : 0.0;
                const unsigned go_w = VideoCommon::ResInfo::out_w.load(), go_h = VideoCommon::ResInfo::out_h.load();
                // the height the picture really takes in the window: the game output fitted to the window by its aspect ratio
                const int area_h = go_w && go_h && win_w > 0 ? int(std::min(win_h, win_w * go_h / go_w) + 0.5) : int(win_h + 0.5);
                m_rows.push_back({QStringLiteral("Res"),
                                  QStringLiteral("%1/%2/%3%4").arg(r_h).arg(go_h).arg(area_h).arg(path),
                                  {}});
            }
            if (nvnh_dlss_info) {   // dlsstoggle (NPARK 11-21): DLSS Super Resolution on or off (the hotkey "Toggle DLSS"), and what it scales
                unsigned d[6] = {}; nvnh_dlss_info(d);
                const QString v = !d[0] ? QStringLiteral("off") : !d[1] ? QStringLiteral("on (idle)")
                                : d[4] != d[2] ? QStringLiteral("%1>%2>%3").arg(d[3]).arg(d[4] * d[3] / (d[2] ? d[2] : 1)).arg(d[5]) : QStringLiteral("%1>%2").arg(d[3]).arg(d[5]);
                m_rows.push_back({QStringLiteral("DLSS"), v, {}});
            }
            {   // fgtoggle (NPARK 11-21): frame generation's state over the presenter's last two-second window (F10 turns it on and off)
                namespace F = VideoCommon::FgStatus;
                const u32 mode = F::mode.load(std::memory_order_relaxed);
                const QString state = mode == F::Off ? QStringLiteral("off")
                                    : mode == F::Waiting ? QStringLiteral("on (idle)")
                                    : mode == F::Optical ? QStringLiteral("optical")
                                    : QStringLiteral("DLSS-G x%1").arg(F::multiplier.load(std::memory_order_relaxed));
                const QString rate = mode == F::Optical || mode == F::Dlssg
                    ? QStringLiteral(" %1>%2").arg(F::original_fps10.load(std::memory_order_relaxed) / 10).arg(F::output_fps10.load(std::memory_order_relaxed) / 10)
                    : QString{};
                m_rows.push_back({QStringLiteral("FG"), state + rate, {}});
            }
        }
    }
    QStringList description{QStringLiteral("FPS: %1").arg(m_fps)};
    if (!m_fps_only && m_show_game) description.append(m_game_fps);
    for (const auto& row : m_rows) {
        description.append(QStringLiteral("%1: %2 %3").arg(row.label, row.value, row.share).trimmed());
    }
    if (m_shader_burst > 0) description.append(QStringLiteral("Shaders: %1").arg(m_shader_burst));
    setAccessibleDescription(description.join(QStringLiteral(", ")));
    // The width only grows while the mode stays the same, so changing numbers never make the
    // box jump narrower and wider
    const QSize panel_size = drawPanel(nullptr);
    m_width_floor = panel_size.width();
    setFixedSize(panel_size);
    resetPosition();
    update();
}

// Lays out the panel and, when a painter is given, draws it. Sizing and painting share this
// so the box always fits what is drawn.
QSize PerformanceOverlay::drawPanel(QPainter* painter) const {
    const QFont& big_font = m_fps_only ? m_compact_font : m_big_font;
    const QFontMetrics big{big_font}, unit{m_unit_font}, label{m_label_font},
        value{m_value_font}, tag{m_tag_font};
    const QString unit_text = QStringLiteral("FPS");
    const bool has_sub = !m_fps_only && (m_show_game || !m_tag.isEmpty());
    const QString sub_gap = QStringLiteral("  ");

    // Sizing uses the same digit cells as painting, so a new value of the same length never needs
    // more room
    int content = std::max(TabWidth(big_font, m_fps_whole), big.horizontalAdvance(QStringLiteral("00"))) +
                  UnitGap + unit.horizontalAdvance(unit_text);
    if (has_sub) {
        const int game = m_show_game ? TabWidth(m_value_font, m_game_fps) : 0;
        const int gap = m_show_game && !m_tag.isEmpty() ? value.horizontalAdvance(sub_gap) : 0;
        content = std::max(content, game + gap + tag.horizontalAdvance(m_tag));
    }
    int label_width = 0;
    int value_width = 0;
    for (const auto& row : m_rows) {
        label_width = std::max(label_width, label.horizontalAdvance(row.label));
        const int share = row.share.isEmpty()
                              ? 0 : TabWidth(m_value_font, QStringLiteral(" ") + row.share);
        value_width = std::max(value_width, TabWidth(m_value_font, row.value) + share);
    }
    const int value_x = Padding + label_width + LabelGap;
    content = std::max(content, label_width + LabelGap + value_width);
    // The spinner row sits below the rows while shaders are being built.
    const QString shader_count = QString::number(m_shader_burst);
    const QString shader_label = QStringLiteral("Shaders");
    if (m_shader_burst > 0) {
        content = std::max(content, SpinnerSize + LabelGap + label.horizontalAdvance(shader_label) +
                                        LabelGap + TabWidth(m_value_font, shader_count));
    }
    const int panel_width = std::max(content + Padding * 2, m_width_floor);

    int y = Padding + big.capHeight();
    if (painter) {
        painter->setFont(big_font);
        painter->setPen(Qt::white);
        // The number is right-aligned in its measured slot so "FPS" stays put as digits change
        const int number_slot = std::max(TabWidth(big_font, m_fps_whole),
                                         big.horizontalAdvance(QStringLiteral("00")));
        DrawTab(*painter, big_font,
                QPointF{Padding + number_slot - TabWidthF(big_font, m_fps_whole), static_cast<qreal>(y)},
                m_fps_whole);
        painter->setFont(m_unit_font);
        painter->setPen(Accent);
        painter->drawText(QPoint{Padding + number_slot + UnitGap, y}, unit_text);
    }
    if (has_sub) {
        y += 8 + value.capHeight();
        if (painter) {
            int x = Padding;
            if (m_show_game) {
                painter->setFont(m_value_font);
                painter->setPen(Qt::white);
                DrawTab(*painter, m_value_font, QPointF(x, y), m_game_fps);
                x += TabWidth(m_value_font, m_game_fps + sub_gap);
            }
            painter->setFont(m_tag_font);
            painter->setPen(Accent);
            painter->drawText(QPoint{x, y}, m_tag);
        }
    }
    y += 8;
    if (painter) {
        painter->setPen(Divider);
        painter->drawLine(Padding, y, panel_width - Padding, y);
    }
    y += 3;
    for (const auto& row : m_rows) {
        if (painter) {
            const QRect line{Padding, y, content, RowHeight};
            painter->setFont(m_label_font);
            painter->setPen(Accent);
            painter->drawText(line, Qt::AlignLeft | Qt::AlignVCenter, row.label);
            painter->setFont(m_value_font);
            int x = value_x;
            if (m_fps_only) {
                x = panel_width - Padding - TabWidth(m_value_font, row.value);
            }
            painter->setPen(Qt::white);
            DrawTabRow(*painter, m_value_font, x, y, row.value);
            if (!row.share.isEmpty()) {
                x += TabWidth(m_value_font, row.value + QStringLiteral(" "));
                painter->setPen(Dim);
                DrawTabRow(*painter, m_value_font, x, y, row.share);
            }
        }
        y += RowHeight;
    }
    if (m_shader_burst > 0) {
        if (painter) {
            drawShaderSpinner(*painter, QRectF(Padding, y + (RowHeight - SpinnerSize) / 2.0,
                                               SpinnerSize, SpinnerSize));
            const int label_x = Padding + SpinnerSize + LabelGap;
            painter->setFont(m_label_font);
            painter->setPen(Accent);
            painter->drawText(QRect{label_x, y, panel_width - Padding - label_x, RowHeight},
                              Qt::AlignLeft | Qt::AlignVCenter, shader_label);
            const int count_x = label_x + label.horizontalAdvance(shader_label) + LabelGap;
            painter->setFont(m_value_font);
            painter->setPen(Qt::white);
            DrawTabRow(*painter, m_value_font, count_x, y, shader_count);
        }
        y += RowHeight;
    }
    return {panel_width, y + Padding - 2};
}

void PerformanceOverlay::drawShaderSpinner(QPainter& painter, const QRectF& area) const {
    const double outer = area.height() / 2.0;
    const double inner = outer * 0.45;
    QPen pen;
    pen.setWidthF(std::max(1.5, outer * 0.26));
    pen.setCapStyle(Qt::RoundCap);
    painter.save();
    painter.setRenderHint(QPainter::Antialiasing);
    painter.translate(area.center());
    for (int i = 0; i < SpinnerSpokes; ++i) {
        // Spoke i sits i*30 degrees clockwise from 12 o'clock. The head (spoke m_spin_step)
        // is opaque; spokes behind it fade, so the bright spoke appears to turn clockwise.
        const int age = (m_spin_step - i + SpinnerSpokes) % SpinnerSpokes;
        pen.setColor(QColor{255, 255, 255, 255 - age * SpinnerAlphaStep});
        painter.setPen(pen);
        painter.drawLine(QPointF{0.0, -inner}, QPointF{0.0, -outer});
        painter.rotate(360.0 / SpinnerSpokes);
    }
    painter.restore();
}

void PerformanceOverlay::paintEvent(QPaintEvent*) {
    QPainter painter{this};
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor{0, 0, 0, UISettings::values.perf_overlay_opacity.GetValue()});
    painter.drawRoundedRect(QRectF{rect()}, Radius, Radius);
    drawPanel(&painter);
}

void PerformanceOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        m_drag_start_pos = event->position().toPoint();
        m_press_global_pos = event->globalPosition().toPoint();
        m_dragging = false;
        event->accept();
        return;
    }
    QWidget::mousePressEvent(event);
}

void PerformanceOverlay::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        if (!m_dragging && (event->globalPosition().toPoint() - m_press_global_pos)
                                   .manhattanLength() < QApplication::startDragDistance()) {
            event->accept();
            return;
        }
        m_dragging = true;
        QWidget* anchor = m_anchor ? m_anchor.data() : m_mainWindow;
        if (anchor) {
            const QPoint desired = event->globalPosition().toPoint() - m_drag_start_pos;
            const QPoint default_position = anchor->mapToGlobal(
                QPoint{anchor->width() - width() - Margin, Margin});
            m_drag_offset = desired - default_position;
            resetPosition();
        }
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void PerformanceOverlay::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (!m_dragging) {
            m_fps_only = !m_fps_only;
            m_width_floor = 0;
            UISettings::values.perf_overlay_fps_only = m_fps_only;
            updateLines(m_native_fps);
        }
        m_dragging = false;
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void PerformanceOverlay::wheelEvent(QWheelEvent* event) {
    // Touchpads send small deltas, so collect them into whole wheel notches.
    m_wheel_delta += event->angleDelta().y();
    const int steps = m_wheel_delta / QWheelEvent::DefaultDeltasPerStep;
    if (steps != 0) {
        m_wheel_delta -= steps * QWheelEvent::DefaultDeltasPerStep;
        const int opacity = std::clamp(
            static_cast<int>(UISettings::values.perf_overlay_opacity.GetValue()) +
                steps * OpacityStep, MinOpacity, MaxOpacity);
        UISettings::values.perf_overlay_opacity = static_cast<u8>(opacity);
        update();
    }
    event->accept();
}

void PerformanceOverlay::closeEvent(QCloseEvent* event) {
    stopSampling();
    emit closed();
    event->accept();
}
