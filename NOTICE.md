# NXN1X — licenses

NXN1X as a whole is distributed under the **GNU General Public License v3.0 or later** (`LICENSE`), because it is built on Eden / yuzu (`source/`, GPL-3.0-or-later and GPL-2.0-or-later files).

Parts keep their own licenses and notices:

| Part | License | File |
|---|---|---|
| `source/` (Eden / yuzu based emulator) | GPL-3.0-or-later (some files GPL-2.0-or-later) | `source/LICENSE.txt`, SPDX headers, `source/LICENSES/` |
| `host/` (NVN host layer, derived from nxnative) | MIT, © 2026 seedds | `host/LICENSE` |
| `host/dm/third_party/*` (Ryujinx, libvpx, stb_vorbis, FLIP, Vulkan headers, Pretendard, …) | their own licenses | each folder's `LICENSE` |
| `source/externals/*`, `source/dist/*` fonts and themes | their own licenses | each folder's `LICENSE` |

MIT-licensed code is used here under the terms of its MIT license, which permits inclusion in a GPL-3.0 work; its copyright notice is kept in `host/LICENSE`.

## Not in this repository

- NVIDIA DLSS / DLSS-G and Streamline runtime DLLs (`nvngx_dlss.dll`, `nvngx_dlssg.dll`, `sl.*.dll`) are NVIDIA's, under NVIDIA's licenses. They are not part of the source. A binary release that ships them must include NVIDIA's license files (`license.txt`, `nvngx_dlss.license.txt`, `ngx-license.txt`, `reflex.license.txt` from the `streamline` folder).
- No Nintendo keys, firmware, system files, saves or game files are included or distributed. Users supply their own legally obtained dumps.
