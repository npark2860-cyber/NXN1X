// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/loader/nup.h"

#include <algorithm>
#include <atomic>   // the booted NUP (NupBootControl)
#include <cstring>

#include "common/logging.h"
#include "common/string_util.h"
#include "core/core.h"
#include "core/file_sys/nup_container.h"
#include "core/file_sys/common_funcs.h"   // GetBaseTitleID (NupBootControl)
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/deconstructed_rom_directory.h"

namespace Loader {

namespace {

using FileSys::NupContainer;

// NACP stores 16 uncompressed language entries; map BCP-47 tags onto them.
struct LanguageSlot {
    std::string_view tag;
    std::size_t index;
};

constexpr std::array<LanguageSlot, 18> LanguageSlots{{
    {"en-us", 0},   {"en-gb", 1},  {"ja", 2},     {"fr", 3},     {"de", 4},
    {"es-419", 5},  {"es", 6},     {"it", 7},     {"nl", 8},     {"fr-ca", 9},
    {"pt", 10},     {"ru", 11},    {"ko", 12},    {"zh-hant", 13}, {"zh-hans", 14},
    {"pt-br", 15},  {"pl", 16},    {"th", 17},
}};

void CopyNameToEntry(FileSys::LanguageEntry& entry, std::string_view name) {
    const std::size_t length = std::min(name.size(), entry.application_name.size() - 1);
    std::memcpy(entry.application_name.data(), name.data(), length);
    entry.application_name[length] = '\0';
}

void CopyDeveloperToEntry(FileSys::LanguageEntry& entry, std::string_view developer) {
    const std::size_t length = std::min(developer.size(), entry.developer_name.size() - 1);
    std::memcpy(entry.developer_name.data(), developer.data(), length);
    entry.developer_name[length] = '\0';
}

std::unique_ptr<FileSys::NACP> BuildNacp(const NupContainer::PackageInfo& base,
                                         const std::string& version) {
    const std::string& effective_title =
        !base.title_name.empty() ? base.title_name : base.source_dump;
    if (effective_title.empty() && base.localized_titles.empty() && base.publisher.empty() &&
        version.empty()) {
        return nullptr;
    }
    auto nacp = std::make_unique<FileSys::NACP>();
    nacp->raw.supported_languages = static_cast<FileSys::SupportedLanguage>(0x3FFFF);

    for (const auto& [lang, name] : base.localized_titles) {
        for (const auto& slot : LanguageSlots) {
            // The uncompressed RawNACP only stores 16 entries; Polish/Thai exist
            // only in the compressed format, so they fall back to the default.
            if (slot.index < 16 && slot.tag == lang) {
                CopyNameToEntry(nacp->raw.language_entries.language_entries[slot.index], name);
            }
        }
    }
    // Slot 0 is en-us; an exact en-us localized title wins over the default title (SPEC 5.1).
    auto& english = nacp->raw.language_entries.language_entries[0];
    if (!effective_title.empty() && english.application_name[0] == '\0') {
        CopyNameToEntry(english, effective_title);
    }
    if (!base.publisher.empty()) {
        for (auto& entry : nacp->raw.language_entries.language_entries) {
            CopyDeveloperToEntry(entry, base.publisher);
        }
    }
    if (!version.empty()) {
        const std::size_t length = std::min(version.size(), nacp->raw.version_string.size() - 1);
        std::memcpy(nacp->raw.version_string.data(), version.data(), length);
        nacp->raw.version_string[length] = '\0';
    }
    nacp->raw.save_data_owner_id = base.title_id;
    nacp->raw.dlc_base_title_id = FileSys::GetAOCBaseTitleID(base.title_id);

    nacp->language_entries.resize(16);
    std::memcpy(nacp->language_entries.data(), nacp->raw.language_entries.language_entries.data(),
                sizeof(nacp->raw.language_entries.language_entries));
    return nacp;
}

bool IsApplicationAddon(u64 program, u64 addon) {
    const u64 base = FileSys::GetBaseTitleID(program);
    const u64 family = FileSys::GetBaseTitleID(addon);
    // Switch 2 Editions also bundle add-ons published for the Switch 1 application.
    return family == base || ((base >> 56) == 4 && (family >> 56) == 1 &&
                              ((base ^ family) << 8) == 0);
}

std::string DecodeVersion(u32 version) {
    if (version == 0) {
        return {};
    }
    return fmt::format("{}.{}.{}", version >> 16, (version >> 8) & 0xFF, version & 0xFF);
}

} // namespace

AppLoader_NUP::AppLoader_NUP(FileSys::VirtualFile file_)
    : AppLoader(std::move(file_)), nup(std::make_unique<NupContainer>(file)) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return;
    }

    const auto base_index = nup->FindBasePackage();
    const auto update_index = nup->FindBestUpdatePackage();

    // The program comes from the base package; a standalone update container
    // (SPEC §4.1.2) carries a complete patched tree and boots on its own.
    std::optional<std::size_t> program_index;
    std::optional<std::size_t> exefs_index;
    if (base_index && nup->GetExeFSDir(*base_index) != nullptr) {
        program_index = base_index;
        exefs_index = update_index && nup->GetExeFSDir(*update_index) != nullptr
                          ? update_index
                          : base_index;
    } else if (update_index && nup->GetExeFSDir(*update_index) != nullptr) {
        LOG_INFO(Loader,
                 "NUP container is a standalone update; booting the update program directly");
        program_index = update_index;
        exefs_index = update_index;
    }
    if (!program_index || !exefs_index) {
        LOG_ERROR(Loader, "NUP container has no package with a usable ExeFS");
        return;
    }

    const auto& program = nup->GetPackages()[*program_index];
    base_title_id = program.title_id;

    NupContainer::PackageInfo meta;
    meta.title_id = program.title_id;
    meta.title_name = program.title_name;
    meta.localized_titles = program.localized_titles;
    meta.source_dump = program.source_dump;
    meta.publisher = program.publisher;
    meta.icon = program.icon;
    meta.localized_icons = program.localized_icons;
    if (base_index && *base_index != *program_index) {
        const auto& base = nup->GetPackages()[*base_index];
        if (!base.title_name.empty()) {
            meta.title_name = base.title_name;
        }
        if (!base.localized_titles.empty()) {
            meta.localized_titles = base.localized_titles;
        }
        if (!base.source_dump.empty()) {
            meta.source_dump = base.source_dump;
        }
        if (!base.publisher.empty()) {
            meta.publisher = base.publisher;
        }
        if (!base.icon.empty() || !base.localized_icons.empty()) {
            meta.icon = base.icon;
            meta.localized_icons = base.localized_icons;
        }
    }
    display_title = meta.title_name;
    for (const auto& [lang, name] : meta.localized_titles) {
        if (lang == "en-us" && !name.empty()) {
            display_title = name;
        }
    }
    has_program = true;

    const auto exefs = nup->GetExeFSDir(*exefs_index);
    if (exefs == nullptr || exefs->GetFile("main") == nullptr ||
        exefs->GetFile("main.npdm") == nullptr) {
        LOG_ERROR(Loader, "NUP package has no usable ExeFS");
        return;
    }
    secondary_loader = std::make_unique<AppLoader_DeconstructedRomDirectory>(exefs, true, false);

    if (update_index) {
        const auto& update = nup->GetPackages()[*update_index];
        display_version = update.display_version.empty() ? DecodeVersion(update.version)
                                                         : update.display_version;
        running_version = update.version;
    } else {
        running_version = program.version;
        display_version = program.display_version.empty() ? DecodeVersion(program.version)
                                                          : program.display_version;
    }
    nacp_file = BuildNacp(meta, display_version);
    if (!meta.icon.empty()) {
        icon_file = std::make_shared<FileSys::VectorVfsFile>(meta.icon, "icon.dat");
    } else {
        for (const auto& [lang, icon] : meta.localized_icons) {
            if (lang == "en-us") {
                icon_file = std::make_shared<FileSys::VectorVfsFile>(icon, "icon_" + lang + ".dat");
            }
        }
        if (icon_file == nullptr && !meta.localized_icons.empty()) {
            icon_file = std::make_shared<FileSys::VectorVfsFile>(
                meta.localized_icons.front().second, "icon.dat");
        }
    }


}

namespace {
// The NUP loader of the program running (set by Load, cleared when that loader goes): the services ask it for the
// control data a NUP boot has no Control NCA for (NupBootControl)
std::atomic<const AppLoader_NUP*> g_booted_nup{nullptr};
} // namespace

AppLoader_NUP::~AppLoader_NUP() {
    const AppLoader_NUP* self = this;
    g_booted_nup.compare_exchange_strong(self, nullptr);
}

std::unique_ptr<FileSys::NACP> NupBootControl(u64 program_id) {
    const AppLoader_NUP* const loader = g_booted_nup.load();
    if (loader == nullptr || loader->nacp_file == nullptr ||
        FileSys::GetBaseTitleID(program_id) != FileSys::GetBaseTitleID(loader->base_title_id)) {
        return nullptr;
    }
    return std::make_unique<FileSys::NACP>(*loader->nacp_file);
}

FileType AppLoader_NUP::IdentifyType(const FileSys::VirtualFile& nup_file) {
    return FileSys::NupContainer::IsNupFile(nup_file) ? FileType::NUP : FileType::Error;
}

AppLoader::LoadResult AppLoader_NUP::Load(Kernel::KProcess& process, Core::System& system) {
    if (is_loaded) {
        return {ResultStatus::ErrorAlreadyLoaded, {}};
    }
    g_booted_nup.store(this);   // (a NUP boot: NupBootControl)
    if (nup->GetStatus() != ResultStatus::Success) {
        return {nup->GetStatus(), {}};
    }
    if (!has_program || secondary_loader == nullptr) {
        return {ResultStatus::ErrorNUPNoBasePackage, {}};
    }

    const auto result = secondary_loader->Load(process, system);
    if (result.first != ResultStatus::Success) {
        return result;
    }

    system.GetFileSystemController().RegisterProcess(
        process.GetProcessId(), base_title_id,
        std::make_shared<FileSys::RomFSFactory>(*this, system.GetContentProvider(),
                                                system.GetFileSystemController()));

    is_loaded = true;
    return result;
}

ResultStatus AppLoader_NUP::VerifyIntegrity(
    std::function<bool(size_t, size_t)> progress_callback) {
    if (nup->GetStatus() != ResultStatus::Success) {
        return nup->GetStatus();
    }
    return nup->VerifyIntegrity(std::move(progress_callback));
}

std::map<u64, FileSys::VirtualFile> AppLoader_NUP::SiblingProgramImages() {
    std::map<u64, FileSys::VirtualFile> images;
    const auto& packages = nup->GetPackages();
    for (size_t i = 0; i < packages.size(); ++i) {
        const auto& p = packages[i];
        if (p.kind != NupContainer::PackageKind::Base || p.title_id == base_title_id ||
            FileSys::GetBaseTitleID(p.title_id) != FileSys::GetBaseTitleID(base_title_id)) continue;
        auto base = nup->GetRomFSDir(i);
        if (!base) continue;
        std::optional<size_t> update;
        for (size_t j = 0; j < packages.size(); ++j) {
            const auto& candidate = packages[j];
            if (candidate.kind == NupContainer::PackageKind::Update &&
                (candidate.title_id == p.title_id || candidate.title_id == (p.title_id | 0x800ULL)) &&
                (!update || candidate.version > packages[*update].version)) update = j;
        }
        auto tree = update ? NupContainer::MergeShadowDirs(base, nup->GetRomFSDir(*update)) : base;
        if (auto image = FileSys::CreateRomFS(tree)) images.emplace(p.title_id, std::move(image));
    }
    return images;
}

std::vector<u64> AppLoader_NUP::AddonIds() const {
    std::vector<u64> ids;
    for (const auto& pkg : nup->GetPackages()) {
        if (pkg.kind == NupContainer::PackageKind::Addon && pkg.romfs_dir != UINT32_MAX &&
            IsApplicationAddon(base_title_id, pkg.title_id)) {
            ids.push_back(FileSys::GetAOCBaseTitleID(base_title_id) + FileSys::GetAOCID(pkg.title_id));
        }
    }
    return ids;
}

FileSys::VirtualFile AppLoader_NUP::OpenAddon(u64 title_id) {
    if (!IsApplicationAddon(base_title_id, title_id)) return {};
    const auto& packages = nup->GetPackages();
    for (size_t i = 0; i < packages.size(); ++i) {
        const auto& pkg = packages[i];
        if (pkg.kind == NupContainer::PackageKind::Addon && IsApplicationAddon(base_title_id, pkg.title_id) &&
            (pkg.title_id == title_id || FileSys::GetAOCBaseTitleID(base_title_id) +
             FileSys::GetAOCID(pkg.title_id) == title_id)) {
            std::scoped_lock lock(addon_mutex);
            if (auto it = addon_images.find(pkg.title_id); it != addon_images.end()) return it->second;
            auto image = FileSys::CreateRomFS(nup->GetRomFSDir(i));
            if (image) addon_images.emplace(pkg.title_id, image);
            if (image) LOG_INFO(Loader, "NUP add-on {:016X} opened as {:016X}", pkg.title_id, title_id);
            return image;
        }
    }
    return {};
}

ResultStatus AppLoader_NUP::ReadRomFS(FileSys::VirtualFile& out_file) {
    const auto base_index = nup->FindBasePackage();
    const auto base_romfs = base_index ? nup->GetRomFSDir(*base_index) : FileSys::VirtualDir{};
    const auto update_index = nup->FindBestUpdatePackage();
    const auto update_romfs =
        update_index ? nup->GetRomFSDir(*update_index) : FileSys::VirtualDir{};
    // A standalone update carries the complete merged tree; a base+update pair
    // overlays the update onto the base (SPEC §4.1.4).
    const auto merged = FileSys::NupContainer::MergeShadowDirs(base_romfs, update_romfs);
    if (merged == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    out_file = FileSys::CreateRomFS(merged);
    if (out_file == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    return ResultStatus::Success;
}

bool AppLoader_NUP::IsRomFSUpdatable() const {
    // The update is already merged into the exposed RomFS.
    return false;
}

ResultStatus AppLoader_NUP::ReadProgramId(u64& out_program_id) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_id = base_title_id;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadProgramIds(std::vector<u64>& out_program_ids) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_ids.push_back(base_title_id);
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadIcon(std::vector<u8>& buffer) {
    if (icon_file == nullptr) {
        return ResultStatus::ErrorNoIcon;
    }
    buffer = icon_file->ReadAllBytes();
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadTitle(std::string& title) {
    if (!display_title.empty()) {
        title = display_title;
        return ResultStatus::Success;
    }
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    title = nacp_file->GetApplicationName();
    if (title.empty()) {
        return ResultStatus::ErrorNoControl;
    }
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadControlData(FileSys::NACP& control) {
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    control = *nacp_file;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NUP::ReadNSOModules(Modules& modules) {
    if (secondary_loader == nullptr) {
        return ResultStatus::ErrorNotInitialized;
    }
    return secondary_loader->ReadNSOModules(modules);
}

} // namespace Loader
