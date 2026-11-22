// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <vector>
#include <QCheckBox>
#include <QComboBox>
#include <QGroupBox>
#include <QVBoxLayout>
#include "common/dlss_fg_setting.h"
#include <qnamespace.h>
#include "common/settings.h"
#include "core/core.h"
#include "qt_common/config/shared_translation.h"
#include "ui_configure_graphics_advanced.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_graphics_advanced.h"
#include "yuzu/configuration/shared_widget.h"

ConfigureGraphicsAdvanced::ConfigureGraphicsAdvanced(
    const Core::System& system_, std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group_, parent), ui{std::make_unique<Ui::ConfigureGraphicsAdvanced>()}, system{system_} {

    ui->setupUi(this);

    Setup(builder);

    SetConfiguration();

    checkbox_enable_compute_pipelines->setVisible(false);
}

ConfigureGraphicsAdvanced::~ConfigureGraphicsAdvanced() = default;

void ConfigureGraphicsAdvanced::SetConfiguration() {
    const bool global = Settings::IsConfiguringGlobal();
    const int target = Settings::dlss_frame_generation.GetValue(global) ? int(Settings::dlss_target_fps.GetValue(global)) : -1;
    const int index = dlss_mode->findData(target);
    dlss_mode->setCurrentIndex(index >= 0 ? index : 1);
    dlss_inherit->setChecked(!global && Settings::dlss_frame_generation.UsingGlobal() && Settings::dlss_target_fps.UsingGlobal());
}

void ConfigureGraphicsAdvanced::Setup(const ConfigurationShared::Builder& builder) {
    auto& normal_layout = *ui->normal_target->layout();

    auto* fg = new QGroupBox(QStringLiteral("NVIDIA DLSS Frame Generation"), this);
#ifndef NXN1X_DLSS_UI
    fg->hide();
#endif
    auto* layout = new QVBoxLayout(fg);
    dlss_inherit = new QCheckBox(tr("Use global settings"), fg);
    dlss_inherit->setObjectName("dlssInherit");
    dlss_inherit->setVisible(!Settings::IsConfiguringGlobal());
    layout->addWidget(dlss_inherit);
    dlss_mode = new QComboBox(fg);
    dlss_mode->setObjectName("dlssOutputTarget");
    dlss_mode->addItem(QStringLiteral("OFF"), -1);
    dlss_mode->addItem(QStringLiteral("Auto"), 0);
    for (int fps : {30, 60, 90, 120}) dlss_mode->addItem(QStringLiteral("%1 FPS").arg(fps), fps);
    layout->addWidget(dlss_mode);
    fg->setEnabled(!system.IsPoweredOn());
    connect(dlss_inherit, &QCheckBox::toggled, this, [this](bool inherit) {
        if (inherit) {
            const int target = Settings::dlss_frame_generation.GetValue(true) ? int(Settings::dlss_target_fps.GetValue(true)) : -1;
            const int index = dlss_mode->findData(target);
            dlss_mode->setCurrentIndex(index >= 0 ? index : 1);
        }
        SetNvnPath(nvn_path);
    });
    apply_funcs.push_back([this](bool powered_on) {
        if (powered_on) return;
        const bool global = Settings::IsConfiguringGlobal();
        Settings::dlss_frame_generation.SetGlobal(global || dlss_inherit->isChecked());
        Settings::dlss_target_fps.SetGlobal(global || dlss_inherit->isChecked());
        if (global || !dlss_inherit->isChecked()) {
            const int target = dlss_mode->currentData().toInt();
            Settings::dlss_frame_generation.SetValue(target >= 0);
            if (target >= 0) Settings::dlss_target_fps.SetValue(static_cast<u32>(target));
        }
    });
    normal_layout.addWidget(fg);

    // fgsource (NPARK 11-21: "shouldn't optical and DLSS FG be selectable?"): which generator makes the frames, applied live (nvn_streamline.cpp reads it at
    // each present): Auto = DLSS-G where its inputs are ready, optical interpolation where it does not generate; DLSS-G only; optical only
    auto* src = new QGroupBox(QStringLiteral("Frame Generation source"), this);
#ifndef NXN1X_DLSS_UI
    src->hide();
#endif
    auto* src_layout = new QVBoxLayout(src);
    dlss_fg_source_box = new QComboBox(src);
    dlss_fg_source_box->setObjectName("dlssFgSource");
    dlss_fg_source_box->addItem(tr("Auto (DLSS-G, optical where DLSS-G does not generate)"), 0);
    dlss_fg_source_box->addItem(tr("DLSS-G only"), 1);
    dlss_fg_source_box->addItem(tr("Optical interpolation only"), 2);
    { const int i = dlss_fg_source_box->findData(int(Settings::dlss_fg_source.GetValue(true))); dlss_fg_source_box->setCurrentIndex(i >= 0 ? i : 0); }
    src_layout->addWidget(dlss_fg_source_box);
    apply_funcs.push_back([this](bool) { Settings::dlss_fg_source.SetGlobal(true); Settings::dlss_fg_source.SetValue(static_cast<u32>(dlss_fg_source_box->currentData().toInt())); });
    normal_layout.addWidget(src);

    // dlsstoggle (NPARK 11-21): NVIDIA DLSS Super Resolution (Ocarina of Time's upscale), saved; the hotkey "Toggle DLSS" turns it on and off while a game runs
    auto* sr = new QGroupBox(QStringLiteral("NVIDIA DLSS Super Resolution"), this);
    auto* sr_layout = new QVBoxLayout(sr);
    dlss_upscale_box = new QCheckBox(tr("Upscale with DLSS where supported (Ocarina of Time)"), sr);
    dlss_upscale_box->setObjectName("dlssUpscale");
    dlss_upscale_box->setChecked(Settings::dlss_upscale.GetValue(Settings::IsConfiguringGlobal()));
    sr_layout->addWidget(dlss_upscale_box);
    apply_funcs.push_back([this](bool) { Settings::dlss_upscale.SetGlobal(true); Settings::dlss_upscale.SetValue(dlss_upscale_box->isChecked()); });
    normal_layout.addWidget(sr);

    // A map will sort the data for us
    std::map<u32, QWidget*> normal_hold{};

    for (auto setting :
         Settings::values.linkage.by_category[Settings::Category::RendererAdvanced]) {
        const auto& key = setting->GetLabel();
        if (key == "mkw_30fps" || key == "mkw_dynamic_fps" || key == "hyrule_30fps" ||
            key == "hyrule_dynamic_fps" || key == "frame_pacing_mode" ||
            key == "native_async_draws" || key == "native_pipeline_wait_ms" ||
            key == "npu_processing_resolution" || key.starts_with("npu_fg_")) {
            continue;
        }
        ConfigurationShared::Widget* widget = builder.BuildWidget(setting, apply_funcs);

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        const auto id = setting->Id();

        normal_hold.emplace(id, widget);

        // Keep track of enable_compute_pipelines so we can display it when needed
        if (setting->Id() == Settings::values.enable_compute_pipelines.Id()) {
            checkbox_enable_compute_pipelines = widget;
        }
    }

    for (const auto& [id, widget] : normal_hold) {
        normal_layout.addWidget(widget);
    }
}

void ConfigureGraphicsAdvanced::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(is_powered_on);
    }
}

void ConfigureGraphicsAdvanced::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureGraphicsAdvanced::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureGraphicsAdvanced::ExposeComputeOption() {
    checkbox_enable_compute_pipelines->setVisible(true);
}

void ConfigureGraphicsAdvanced::SetNvnPath(bool nvn) {
    nvn_path = nvn;
    dlss_mode->setEnabled(nvn && !dlss_inherit->isChecked());
    dlss_mode->setToolTip(nvn ? tr("Auto prioritizes original frames and targets the current display refresh rate. FPS presets set the output target.")
                                 : tr("Select NCE, NVN and Vulkan to use DLSS Frame Generation."));
}
