// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 SnapRYU contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <limits>
#include <vector>
#include <QFont>
#include <QPointer>
#include <QStringList>
#include <QTimer>
#include <QWidget>
#include "video_core/presentation_stats.h"

namespace VideoCore { class ShaderNotify; }
namespace Core { struct PerfStatsResults; }
class MainWindow;
class QPainter;
class QWheelEvent;

class PerformanceOverlay : public QWidget {
    Q_OBJECT
public:
    explicit PerformanceOverlay(MainWindow* parent = nullptr, QWidget* anchor = nullptr);
    ~PerformanceOverlay() override;
    void UpdateRendererStats(const VideoCore::RendererPerformanceStats& stats);
    void SetDriverText(const QString& text);

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void paintEvent(QPaintEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    void resetPosition(const QPoint& pos = {});
    void updateStats(const Core::PerfStatsResults& results, const VideoCore::ShaderNotify& shaders);
    void updateLines(double native_fps);
    QSize drawPanel(QPainter* painter) const;
    void stopSampling();
    void drawShaderSpinner(QPainter& painter, const QRectF& area) const;

    MainWindow* m_mainWindow = nullptr;
    QPointer<QWidget> m_anchor;
    QPoint m_drag_offset;
    QPoint m_drag_start_pos;
    QPoint m_press_global_pos;
    QFont m_big_font;
    QFont m_compact_font;
    QFont m_unit_font;
    QFont m_label_font;
    QFont m_value_font;
    QFont m_tag_font;
    struct Row {
        QString label;
        QString value;
        QString share;
    };
    std::vector<Row> m_rows;
    QString m_fps{QStringLiteral("--")};
    QString m_fps_whole{QStringLiteral("--")};
    QString m_game_fps;
    QString m_tag;
    QString m_driver;
    std::int64_t m_vram_used = -1;
    std::int64_t m_vram_budget = -1;
    double m_native_fps = std::numeric_limits<double>::quiet_NaN();
    int m_wheel_delta = 0;
    VideoCore::PresentationRateSampler m_present_sampler;
    bool m_sampling = false;
    bool m_dragging = false;
    bool m_show_game = false;
    bool m_fps_only = false;
    int m_width_floor = 0;
    // Shader spinner: shown while shaders are being built. Burst = shaders started since
    // the spinner appeared. m_shaders_last_started < 0 means no sample taken yet.
    QTimer m_spin_timer;
    int m_spin_step = 0;
    int m_shaders_last_started = -1;
    int m_shader_burst_base = -1;
    int m_shader_burst = 0;

signals:
    void closed();
};
