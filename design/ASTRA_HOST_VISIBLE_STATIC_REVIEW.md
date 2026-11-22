# NXN1X · 드라이버 할당 HOST_VISIBLE 메모리를 게스트 원본으로 쓰는 설계

## 결론

1. **전체 게임 메모리에 대한 판정은 ‘현재 구조에서 차단’이다.** 일반 Vulkan 할당 하나에 버퍼를 바인딩하고 CPU가 그 할당의 매핑 포인터로 접근하는 방식 자체는 성립한다. 그러나 현재 NCE에서 게스트가 사용할 주소는 프로세스 페이지 테이블과 Windows 매핑이 함께 결정한다. 드라이버가 주소를 먼저 골라도, 나중에 게임이 같은 페이지를 두 번째 주소에 매핑할 권리는 사라지지 않는다.
2. **결정적인 지점은 세 곳이다.** `KPageTableBase::SetHeapSize`는 이미 배치한 힙의 시작 주소를 돌려준다. `HostMemory::Impl::MapView`는 게스트가 지정한 주소에 기존 물리 페이지를 section handle과 offset으로 매핑한다. `nvnMemoryPoolBuilderSetStorage`는 게임이 이미 소유한 포인터를 받는다. 풀 등록에서 할당자만 바꾸면 NCE가 원래 포인터로 보는 바이트와 GPU 버퍼의 바이트가 분리된다. [힙](../source/src/core/hle/kernel/k_page_table_base.cpp) · [Windows 매핑](../source/src/common/host_memory.cpp) · [NVN 풀](../host/dm/gfx/nvn/hle_nvn.c)
3. **제약의 출처가 다르다.** 256 GiB 단일 arena, GPU를 나중에 만드는 순서, 선형 물리 backing은 NXN1X의 구현 선택이다. 주소 폭, 공개된 포인터, 요청한 목적지 VA, 공유 페이지의 내용·권한·수명을 보존하는 것은 게스트에 제공하는 계약이다. 일반 `vkMapMemory`에는 지정 주소의 두 번째 Windows view를 만들거나 원본 section handle을 얻는 계약이 없다. 드라이버 내부의 미공개 동작은 미확인이다.
4. **최소 변경 범위는 적용 대상에 달려 있다.** 단일 주소를 보장하는 전용 버퍼에 한정하면 최초 할당자, 게스트 페이지 등록과 NCE 주소 배치, NVN의 버퍼 조회 및 제출 수명을 연결해야 한다. 전 DRAM에 적용하려면 여기에 GPU 초기화·종료 순서, PA→CPU 포인터 변환, 고정 VA alias·부분 해제·권한 처리까지 바꿔야 한다. 현 공개 경로로 임의 alias까지 보존하는 일반 non-exportable 할당 해법은 확인되지 않았다.
5. **공통화할 수 있는 범위**는 처음부터 소유권을 선택할 수 있고, 드라이버 포인터가 합법적인 게스트 VA이며, 전체 수명 동안 단일 CPU view와 연속 GPU 바인딩을 보장하는 버퍼다. **별도 처리가 필요한 범위**는 일반 힙에서 뒤늦게 등록한 풀, 여러 VA가 같은 PA를 가리키는 페이지, 비연속 bridge, 가상 풀, 코드·스택, native image와 그래픽 서버 메모리다. 현재 게임의 일반 풀이 자동으로 첫 번째 조건을 만족한다는 근거는 없다.
6. **성능은 연산별로만 확인됐다.** 저장된 N1X 결과에서 드라이버 할당 HOST_VISIBLE의 반복 정점 fetch는 device-local과 비슷했다. 분산 atomicAdd는 1.7~2.0배, 연속 SSBO store는 2.7~2.8배의 시간이 걸렸다. NCE 원본 힙으로 사용한 게임 FPS, CPU 부하, alias·보호 호환성, import 병목 전체의 제거는 미확인이다.

## 검사 결과표

| 검사 항목 | 판정 | 파일·함수 근거 | 필요한 변경 또는 미확인 조건 |
|---|---|---|---|
| 1. 원본 생성과 포인터 공개 | 전체 DRAM은 현재 순서에서 차단. GPU 생성 후 요청되는 전용 영역은 조건부 가능 | `System::Impl::Initialize/Load/SetupForApplicationProcess`, `DeviceMemory::DeviceMemory`, `SetHeapSize`, `nvnMemoryPoolInitialize` | 전 DRAM은 커널이 RAM을 쓰기 전 같은 Vulkan device가 준비돼야 한다. 늦은 전용 할당은 포인터가 공개되기 전 소유자를 정해야 한다. |
| 2. NCE 주소 | 현재 arena 밖의 임의 포인터는 수용 불가 | 로더의 `EnableDirectMappedAddress`, `KProcess::LoadFromMetadata`, `HostMemory::Impl::AdjustMap`, `WindowsNceEnterGuest` | 실제 반환 VA, 유효 비트, 주소 충돌, 정렬, 향후 힙 성장 공간이 모두 맞아야 한다. 비연속 arena 채택 시 커널·로더·페이지 조회도 변경해야 한다. |
| 3. alias·재매핑·보호 | 임의 guest alias를 일반 할당 하나로 처리하는 경로가 없다 | `KPageTableBase::MapMemory/MapCodeMemory`, `KSharedMemory::Map`, `KTransferMemory::Map`, `HostMemory::Impl::MapView/UnmapOnePlaceholder/Protect` | 공개 주소를 유지하면서 동일 페이지의 두 번째 view와 권한을 제공할 수 있어야 한다. 단일 view의 Windows 페이지 보호에 대한 드라이버 지원도 미확인이다. |
| 4. Vulkan 할당 | 일반 host-visible 버퍼의 생성·매핑은 구조상 가능 | `POOL_USAGE`, `pool_shadow_allocate`, `pool_zero_copy_at0`, 저장된 Vulkan 1.4 장치 기록 | 해당 buffer의 요구조건·허용 타입·전용 할당 필요 여부를 따라야 한다. 게임 RAM 크기 및 전체 usage 조합의 실할당은 미확인이다. |
| 5. NVN 주소·raw BDA | 연속 범위의 기존 주소 변환은 재사용 가능. 비연속 범위는 별도 처리 | `nvnBufferGetAddress`, `vk_buf_at`, `pool_gpu_addr`, `vk_gmem_map`, `ContentResolve/StitchRuns` | guest VA→원본 allocation·offset→buffer offset/BDA로 연결한다. alias가 바뀌면 세대·테이블·descriptor를 갱신한다. |
| 6. 수명·동기화 | 같은 바이트의 사본 관리는 제거 가능. CPU/GPU 순서와 결과 공개는 계속 필요 | `nvnMemoryPoolFinalize`, `mtl_pool_del`, `vk_defer_free_buffer`, `pc_readback`, `s56_land`, 완료 스레드의 `g_done` | pool 등록 수명과 원본 allocation 수명을 분리한다. 마지막 GPU 접근·CPU 사용자·alias 해제 후 반환하고 비일관 타입은 flush/invalidate까지 연결한다. |
| 7. 성능 | 지정된 마이크로벤치 연산만 확인. 공용 원본의 종단 간 성능은 자료 부족 | `membench2.c`, `r7.c`, 각 결과 파일 및 기존 게임 로그 | 메모리 타입 번호, vertex 결과, 기존 copy/import 게임 FPS를 전체 게임 메모리 성능으로 일반화할 수 없다. |

## 1. 실제 할당 순서

| 시점 | 코드가 결정하는 것 |
|---|---|
| 시스템 초기화 | `System::Impl::Initialize`에서 `device_memory.emplace()`. `DeviceMemory`는 설정에 맞는 DRAM 크기와 VA 예약 크기로 `Common::HostMemory`를 구성한다. Windows 구현은 실행 권한을 줄 수 있는 페이지 파일 section, 전체 backing의 RW view, NCE용 별도 placeholder를 만든다. 물리 주소 기준은 `DramMemoryMap::Base = 0x80000000`. HLE의 `GetPointer`는 backing base에 PA 오프셋을 더하고 역변환도 같은 연속 영역을 전제한다. [초기화](../source/src/core/core.cpp) · [DRAM 생성](../source/src/core/device_memory.cpp) · [물리 주소 변환](../source/src/core/device_memory.h) |
| 타이틀 선택과 로딩 | `Load`는 타이틀의 메모리 설정을 반영하고 `InitializeKernel → CreateApplicationProcess`를 수행한다. Switch 2의 12 GiB 모드에는 그래픽 서버용 2 GiB가 추가될 수 있다. NCE 로더는 arena 기준 주소를 프로세스 레이아웃에 반영한 뒤 모듈을 적재한다. 따라서 게임의 첫 CPU 명령 전에 이미 물리 페이지와 프로그램 VA를 사용한다. [로드 순서](../source/src/core/core.cpp) · [메모리 크기](../source/src/core/hle/kernel/board/nintendo/nx/k_system_control.cpp) · [NSO 로더](../source/src/core/loader/deconstructed_rom_directory.cpp) |
| GPU 생성 | 애플리케이션 프로세스 생성 후 `SetupForApplicationProcess → VideoCore::CreateGPU → CreateRenderer → RendererVulkan::device(CreateDevice(...))`. NVN은 `NvnBridge::Attach`로 이 장치와 스케줄러를 전달받는다. CPU manager 초기화는 그 뒤다. 따라서 실행 중의 힙 확장 때에는 GPU가 있어도 초기 DRAM을 할당할 때에는 현재 렌더러의 device가 없다. 다른 임시 device에서 만든 `VkDeviceMemory`를 그대로 렌더 장치에 바인딩하는 설계도 성립하지 않는다. [GPU 생성](../source/src/video_core/video_core.cpp) · [렌더러](../source/src/video_core/renderer_vulkan/renderer_vulkan.cpp) · [NVN 장치 연결](../source/src/video_core/renderer_vulkan/nvn_bridge_device.cpp) |
| 게임 힙 확장 | `Svc::SetHeapSize → KPageTableBase::SetHeapSize → MemoryManager().AllocateAndOpen(KPageGroup) → Operate → Memory::MapMemoryRegion`. 성장분의 물리 페이지를 얻어 이미 정해진 `m_current_heap_end`에 붙이고 `m_heap_region_start`를 반환한다. SVC 크기 정렬은 2 MiB다. 다음 성장 호출이 앞서 반환한 포인터를 무효화해서는 안 된다. 게임의 힙 내부 suballocation은 호스트가 어느 바이트가 NVN용인지 결정하는 지점이 아니다. [SVC](../source/src/core/hle/kernel/svc/svc_physical_memory.cpp) · [힙 매핑](../source/src/core/hle/kernel/k_page_table_base.cpp) · [물리 페이지 할당](../source/src/core/hle/kernel/k_memory_manager.cpp) |
| NVN 등록과 Vulkan 바인딩 | `nvnMemoryPoolBuilderSetStorage(m,n)`가 호출자 포인터를 저장한다. 일반 풀의 `nvnMemoryPoolInitialize`는 이를 `mtl_pool_add`와 `nvn_dirty_add_range`에 등록한다. Windows 백엔드는 view/chunk 경계를 나누고 lazy 모드에서는 첫 GPU 조회 때 `pool_make_0`에서 import 또는 사본을 만든다. 등록 시점에는 게임이 이미 같은 포인터를 저장했거나 데이터를 채웠을 수 있다. [NVN 등록](../host/dm/gfx/nvn/hle_nvn.c) · [풀 및 바인딩](../host/dm/windows/gfx_vk_core.c) |

모든 GPU 메모리가 일반 게임 힙을 거치지도 않는다. Switch 2 `AllocateGpuOnlyBacking`은 커널 메모리 풀에서 독립적으로 page group을 얻어 애플리케이션에 `IoMemory`로 매핑한다. 이름은 GPU-only여도 현 구현은 매핑 주소를 보관한다. `h_ni_texture`는 외부 picture handle에서 얻은 메모리를 나중에 NVN 풀로 감싼다. 가상 NVN 풀은 최초 storage가 null일 수 있고 `vpool_reserve/MapVirtual`로 별도 주소와 물리 페이지를 연결한다. 이 범위를 단순히 heap allocator에서 가로채면 빠진다. [그래픽 서버 할당](../source/src/core/hle/service/nx2_gsv.h) · [native image와 가상 풀](../host/dm/gfx/nvn/hle_nvn.c)

## 2. 드라이버 주소를 먼저 채택하는 경우

**단일 주소 예.** 드라이버가 돌려준 `P`가 게스트 주소 폭에 들어가고 충돌이 없다고 가정한다. 페이지 테이블이 해당 allocation과 offset을 가리키고, NCE가 `P+i`로 직접 읽고 쓰며 GPU가 같은 메모리의 `VkBuffer` 오프셋 `i`를 사용하면 저장소는 하나다. 이 구성에는 import나 고정 주소 Vulkan 매핑이 필요하지 않다. 저장 벤치가 확인한 것은 CPU의 매핑 포인터와 Vulkan 버퍼의 이 관계까지다. **게스트 페이지 관리자에 편입되고 NCE로 실행된 사례는 확인되지 않았다.**

**현재 구현과 부딪히는 지점은 두 가지다.**

- **주소 배치.** `KPageTableBase::GetAddressSpaceWidth`는 프로그램 설정에 따라 32/36/39비트를 구분한다. 39비트 프로세스의 NCE Map39 창은 256 GiB로 축소되며 현재 표에서 heap 16 GiB, alias 64 GiB, stack 2 GiB의 영역을 배치한다. Windows는 256 GiB arena에 정렬 여유를 추가해 2 MiB 정렬로 예약한다. 선호 구간은 `[2^36, 2^39)`이고 실패 시 하한만 완화한다. 로더는 이 기준과 프로그램별 코드 오프셋을 조합한다. 예컨대 39비트 코드 배치에 쓰는 `0x80000000 + ASLR offset`은 물리 DRAM의 동명 숫자와 다른 계산이다. 이 선호 하한과 arena 크기는 NXN1X 정책이고, 게스트에게 보고한 폭·영역·이미 반환한 주소는 지켜야 한다. NCE의 실제 SP와 분기는 이 VA를 네이티브로 사용한다. [영역 표](../source/src/core/hle/kernel/k_address_space_info.cpp) · [프로세스 레이아웃](../source/src/core/hle/kernel/k_process.cpp) · [NRO 진입](../source/src/core/loader/nro.cpp) · [NCE 진입 어셈블리](../source/src/core/arm/nce/windows_nce_entry.asm)
- **예약의 소유권.** 기존 arena 안은 이미 placeholder로 예약돼 있고 일반 `vkMapMemory`의 반환 주소는 호출자가 지정하지 않는다. 순서를 바꿔 `P`를 먼저 얻어도 이미 매핑된 `P`를 통째로 포함하는 placeholder 예약은 그대로 할 수 없다. 주소 폭에 들어간다는 조건만으로도 부족하다. 현재 `AdjustMap`은 arena 밖을 잘라 내므로 바깥 포인터를 페이지 테이블에만 넣어서는 OS 매핑·보호·해제가 일치하지 않는다. 단일 범위 후보라도 기존 예약에서 분리된 구간을 표현하고 충돌을 피하도록 레이아웃을 바꿔야 한다. 621.24의 일반 매핑이 충분히 낮은 주소에서 필요한 크기·정렬·인접 여유를 반복 보장한다는 자료는 없다. [HostMemory 예약과 AdjustMap](../source/src/common/host_memory.cpp)

## 3. 두 VA에서 같은 바이트가 보여야 하는 경우

예를 들어 A에서 얻은 페이지를 게임이 B에 매핑한다고 하자. B에서 쓴 값은 허용된 동기화 이후 A와 GPU에도 보여야 하고, B만 해제해도 A는 유효해야 한다. 권한도 view별로 달라질 수 있다. 처음 선택한 A가 드라이버 주소와 정확히 같아도 이 조건은 남는다.

- `MapMemory/UnmapMemory`와 `MapCodeMemory`는 원본 범위의 페이지를 다른 VA에 매핑하거나 권한을 변경한다. `KSharedMemory::Map`은 같은 page group을 대상 프로세스의 지정 VA에, `KTransferMemory::Initialize/Map/Unmap`은 원래 페이지를 잠근 뒤 지정 VA에 매핑한다. 가상 주소만 해제하는 일과 물리 페이지를 반환하는 일이 분리돼 있다. [페이지 테이블](../source/src/core/hle/kernel/k_page_table_base.cpp) · [shared memory](../source/src/core/hle/kernel/k_shared_memory.cpp) · [transfer memory](../source/src/core/hle/kernel/k_transfer_memory.cpp)
- `Memory::MapMemoryRegion`은 소프트웨어 페이지표를 갱신하고 Windows fastmem에 `(guest VA, PA − DRAM base, size, permissions)`를 전달한다. 호스트 측은 4 KiB 단위 검사를 하고 지정된 placeholder에 `MapViewOfFile3(backing_handle, ..., target, offset, ...)`를 수행한다. 부분 해제는 `UnmapViewOfFile2`와 잔여 view 재구성이며 보호 변경에는 `VirtualProtect`와 실행 코드의 I-cache 처리가 있다. NCE 스택에는 조건부 private lease/복구 경로도 있다. `P`를 아는 것만으로 section offset, 다른 view, 독립 권한, 실행 가능한 코드 매핑이 생기지 않는다. 일반 드라이버 매핑을 임의로 부분 unmap하거나 RX/guard로 바꿔도 드라이버 계약과 NCE가 유지되는지는 미확인이다. [Memory](../source/src/core/memory.cpp) · [Windows view 관리](../source/src/common/host_memory.cpp) · [NCE 스택 처리](../source/src/core/arm/nce/arm_nce_windows.cpp)
- 로컬 Vulkan 규정상 이미 매핑된 같은 memory를 `vkMapMemory`로 다시 매핑할 수 없고, Win32 handle을 꺼내려면 최초 할당 시 해당 export handle type을 지정해야 한다(`00678`, `00662`). 따라서 **현재의 Windows section-view 방식을 유지하려면** 애초에 exportable로 생성하거나 기존 section을 import하는 등 외부 메모리 계약이 필요하다. 드라이버가 평범한 할당에 대해 곧바로 section을 제공한다는 경로는 없다. 반면 alias가 전혀 필요 없는 버퍼에는 이 결론이 적용되지 않는다. GPU sparse binding만으로는 두 번째 **CPU** VA가 만들어지지 않는다. [로컬 Vulkan 유효 사용 규정](../tools/cpm/vulkanheaders/v1.4.345/registry/validusage.json) · [export와 Windows view를 비교한 기존 벤치](../membench2/membench2.c)

## 4. Vulkan 할당과 GPU 주소 연결

**기록된 장치는 NVIDIA RTX Spark N1X, 드라이버 621.24, native Vulkan이다.** 같은 정보 파일의 Microsoft D3D12 어댑터 행은 제외했다. heap 1의 type 2는 `HOST_VISIBLE | HOST_COHERENT`(`0x6`), type 3은 여기에 `HOST_CACHED`가 추가된다(`0xe`). heap 0의 type 1은 `DEVICE_LOCAL`. `minMemoryMapAlignment=64`, `nonCoherentAtomSize=64`, UBO/SSBO offset 최소 정렬은 각각 64/16바이트다. `minImportedHostPointerAlignment=4096`과 기존 host pointer의 허용 비트 `0xc`는 **import 입력**에 대한 값이다. 일반 buffer의 바인딩 정렬·허용 비트는 그 buffer의 `VkMemoryRequirements`로 결정한다. [저장된 장치 정보](../build/dlss-vulkaninfo.txt) · [기존 장치 보고서](../nvidia_report/README.md)

일반 원본 후보에는 `vkCreateBuffer(POOL_USAGE) → vkGetBufferMemoryRequirements → vkAllocateMemory → vkBindBufferMemory → vkMapMemory`를 사용할 수 있다. 실제 `pool_shadow_allocate`는 일반 create info로 buffer를 만들고 허용 비트 안에서 cached host-visible, 이어 coherent host-visible을 찾는다. 이 함수의 **할당 방식**만 근거로 사용했다. 현재 이 함수를 부르는 시점과 소유권은 사본용이다. 반면 `pool_zero_copy_at0/s48`은 외부 buffer create info와 host-pointer import를 사용한다. 일반 분기에 export/import 구조체가 암묵적으로 추가되는 코드는 없다. Vulkan 드라이버 내부 배치·캐시 정책은 소스로 확인할 수 없다. [할당 분기와 POOL_USAGE](../host/dm/windows/gfx_vk_core.c)

- pool buffer는 vertex/index, indirect, uniform/storage buffer와 texel buffer, transfer src/dst를 지원하고 기능 사용 시 XFB와 shader device address가 추가된다. BDA에는 device 기능, buffer usage와 `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`가 필요하다. allocation은 `req.size` 이상, 바인딩 offset은 `req.alignment` 배수여야 한다. 공통 arena에서 여러 buffer를 쓰면 각 리소스의 비트 교집합·offset·전용 할당 조건을 만족시키거나 별도 소유해야 한다. 기록상 `maxBufferSize=1 TiB`, `maxStorageBufferRange=0xffffffff`이므로 큰 원본 하나를 만들 수 있다는 가정과 전체를 단일 SSBO descriptor로 쓸 수 있다는 가정도 구분해야 한다. 게임 크기·usage 조합의 일반 buffer에 대한 실측 mask, 실제 최대 할당 성공 크기 및 CPU 페이지 속성은 미확인이다.
- **실제 예외가 기록돼 있다.** 기존 N1X bridge의 외부 sparse buffer 조사에서는 블록 64 KiB, `memoryTypeBits=0x2`여서 host-visible coherent 후보가 없었다. 장치의 `sparseBinding=true`만 보고 모든 비연속 원본을 연결할 수 없다. 이 로그는 해당 외부 buffer 조합의 결과이며, 일반 non-external sparse buffer의 허용 비트는 미확인이다. [bridge 조사 로그](../logs_poolcopy/selftest_copy_154230.txt) · [sparse buffer 생성 조건](../host/dm/windows/gfx_vk_core.c)
- **주소의 숫자 관계.** HLE의 `nvnMemoryPoolGetBufferAddress`는 guest 포인터, `nvnBufferGetAddress`는 guest pool base + buffer offset을 반환한다. CPU가 shader/descriptor를 해석하는 `mtl_addr`도 guest 주소를 직접 사용한다. Vulkan은 `vk_buf_at`로 `VkBuffer + (guest VA − chunk.lo)`를 구하고, `pool_gpu_addr`가 실제 BDA를 조회한다. `vk_gmem_map`은 guest 범위마다 `{lo, hi, gpuBase − guestBase}`를 적는다. 셰이더가 guest 포인터에 delta를 더하므로 CPU 포인터와 실제 BDA의 숫자가 같을 필요는 없다. 표는 4,096칸 중 0번이 예약돼 있다. 물리 run마다 항목을 무제한 늘릴 수 없다. [HLE 주소](../host/dm/gfx/nvn/hle_nvn.c) · [Vulkan lookup](../host/dm/windows/gfx_vk_core.c) · [셰이더 표 형식](../host/dm/gfx/nvn/dmsh_abi.h)
- **공용 원본 도입 시 바뀌는 부분.** 같은 allocation을 가리키는 각 guest VA를 동일 물리 offset에 연결하되, 하나의 큰 VkBuffer를 공유하면 `vk_buf_at`과 BDA delta 양쪽에 suballocation offset을 반영해야 한다. remap 시 풀·descriptor·raw 표 세대도 함께 갱신해야 한다. 현재 `ContentResolve/ImportAlias/StitchRuns`는 `BackingBasePointer` 기준으로 물리 run을 확인하거나 Windows view를 이어 붙인다. 연속 guest VA가 비연속 GPU memory를 가리키면 raw 주소별 변환만으로 고정 기능 vertex/index 또는 단일 descriptor의 연속 범위를 충족하지 못한다. 연속 backing, 지원되는 sparse GPU 매핑, 또는 해당 범위에 한정된 bridge 사본이 필요하다. [backing 조회 및 run 결합](../source/src/core/nvn/nvn_layer.cpp)

## 5. 수명과 동기화

**제거 가능한 비용.** 동일한 바이트가 처음부터 CPU와 GPU의 원본이라면 그 buffer를 위해서만 존재하는 CPU→shadow 비교·업로드, `pc_upload`, device-local→readback, `pc_merge/pc_merge_async`, 양쪽 값의 충돌 병합과 사본을 위한 volatile/dirty 추적은 제거할 수 있다. 일반 allocation을 다시 host pointer로 import하면 이 설계의 이점이 없어지므로 canonical allocation의 원래 buffer를 직접 조회해야 한다. [현재 사본 동기화](../host/dm/windows/gfx_vk_core.c)

**유지할 기능.** CPU가 마지막 쓰기를 마친 뒤 GPU 소비가 시작되는 순서, GPU 쓰기 이후 host read 가시성과 완료 관측, queue 간 의존성, event/query/counter 공개 순서를 보존해야 한다. coherent는 해당 매핑의 명시적 flush/invalidate 부담을 없애지만 동시 쓰기 순서나 게스트의 atomic 의미를 보장하지 않는다. 비일관 타입이면 메모리 객체 기준 offset과 atom 경계에 맞춰 flush/invalidate하고, CPU가 GPU 결과를 보려면 완료 뒤 invalidate한다. 현 `nvnMemoryPoolInvalidateMappedRange`는 로그 위주이므로 이 경로를 그대로 비일관 원본에 쓸 수 없다. 이미지 tiling·format 변환, texture write watch와 s56 render-target 착지, sparse 주소 추적, shader/descriptor 무효화도 이유가 별개다. 저장 장치 정보에서 type 2/3은 optimal-tiled image용으로 표시되지 않는다. [NVN flush/invalidate](../host/dm/gfx/nvn/hle_nvn.c) · [완료 스레드와 s56](../host/dm/windows/gfx_vk_core.c) · [메모리 타입별 image 지원](../build/dlss-vulkaninfo.txt)

**소유자 교체가 필요한 지점.** 현재 `nvnMemoryPoolFinalize`는 replay를 drain하고 alias·watch를 해제한 뒤 `mtl_pool_del`을 호출한다. 원본 DRAM은 게임이 계속 보유할 수 있다. 한 NVN pool의 finalize나 chunk 축출이 원본의 `vkUnmapMemory/vkFreeMemory`로 이어지면 안 된다. allocation에는 guest 물리 페이지·view·NVN 등록·BDA 표와 진행 중인 GPU 제출의 수명을 따로 귀속시켜야 한다. 빈 VA나 같은 PA가 재사용될 때 이전 제출이 새 데이터에 접근하지 않도록 세대와 실제 마지막 사용을 확인한다. `vkFreeMemory`는 해당 memory를 참조하는 모든 제출 완료가 전제다(`00677`). 현 shutdown은 GPU를 정리한 뒤 kernel을 내리고 경우에 따라 DeviceMemory를 다음 실행까지 유지한다. 공용 원본 소유 device는 마지막 CPU 포인터·GPU 사용과 원본 해제까지 살아 있어야 한다. [풀 해제와 지연 반환](../host/dm/windows/gfx_vk_core.c) · [시스템 종료 순서](../source/src/core/core.cpp) · [로컬 Vulkan 규정](../tools/cpm/vulkanheaders/v1.4.345/registry/validusage.json)

## 6. 기존 성능 자료의 적용 범위

아래 시간은 **기존 파일에 저장된 GPU 측정값**이다. 서로 다른 세션의 절대 시간을 나눠 배속을 산출하지 않았다.

| 실제 측정한 접근 | 저장된 결과 | 해석 |
|---|---|---|
| 인덱스와 정점 buffer의 고정 기능 fetch | membench2의 39×1399 draw에서 일반 driver host 8.7 µs, device-local 9.8 µs, exportable driver host 438.1 µs, export한 view로 CPU가 쓴 경우 445.2 µs, 기존 section의 Win32 import 1,133.3 µs | 일반 할당과 외부 메모리 경로의 차이가 이 draw에서 확인됐다. exportable에서 이미 느렸고, view 양방향 데이터 확인도 별도로 기록됐다. |
| 같은 160 KB vertex buffer를 99 draw에서 재사용 | R7의 각 세션 내 local 대비 import 15.7배, 일반 type 2와 3 각각 약 1.0배의 시간 | 반복 정점 fetch 결과. 임의 shader read나 CPU 캐시 성능 측정이 아니다. |
| 1 MB 범위로 분산한 400만 atomicAdd | local 대비 import 21.6배, 일반 type 2 2.0배, type 3 1.7배 | 일반 HOST_VISIBLE atomic에도 비용이 남는다. CPU/GPU 간 동시 atomic이나 특정 게임의 동일 주소 집중 쓰기는 이 측정 범위 밖이다. |
| 16 MB 연속 SSBO store | local 대비 import 1.5배, 일반 type 2 2.8배, type 3 2.7배 | 일반 할당이 모든 접근에서 import보다 빨라진다는 예측은 성립하지 않는다. 서로 다른 실행 간 절대 순위는 확정하지 않는다. |

`membench2.c`에는 일반 할당, exportable, host-pointer/Win32 import 분기가 구분돼 있다. R7 원자료는 N1X에서 각각의 buffer 요구 비트 확인 후 할당한 일반 type 2/3과 device-local을 9회 교차 측정한 것이다. R7은 게임의 전체 `POOL_USAGE+BDA` 및 멀티 GiB 힙 구성 시험이 아니다. `membench3/4`의 shader·정점 형식과 pull 시험, `membench5`의 import/device-local 시험은 각기 다른 접근을 측정한다. 게임의 기존 copy/import A/B에는 별도의 guest RAM이 있었으므로 공용 원본 모드의 프레임 시간으로 사용할 수 없다. [membench2 소스](../membench2/membench2.c) · [membench2 결과](../membench2/results.txt) · [R7 소스](../build/gpu-cache-review/r7.c) · [import](../build/gpu-cache-review/type0.txt) · [type 2](../build/gpu-cache-review/type2.txt) · [type 3](../build/gpu-cache-review/type3.txt) · [기존 게임 copy 로그](../logs_poolcopy/ab_copy_160639.txt)

## 적용 범위에 대한 최종 판단

일반 할당의 포인터를 **처음부터** 채택하는 발상은 단일 CPU view의 버퍼에서 유효하다. 게임의 표준 힙으로 넓히려면 최초 할당 때는 알 수 없는 미래의 alias·transfer·보호 요청까지 처리할 수 있어야 한다. 어느 시점에 그런 요청이 오면 문자열이나 일반 구조체처럼 이미 노출된 포인터를 일괄 교체할 수 없고, GPU BDA와 진행 중인 명령도 그대로 남는다. 따라서 현재 코드에서 가장 작은 성립 단위는 **소유권·주소·향후 매핑 계약을 최초 할당자가 통제하는 버퍼**다. 표준 게임 RAM 전체의 성립 조건은 GPU 속도보다 **같은 물리 페이지를 요구된 모든 NCE 주소에서 보이게 할 방법**이다.

이번 작업은 소스, 기존 설계, 저장된 벤치 및 로그를 읽은 정적검사다. 변경한 파일은 이 문서 하나다.

## 7. 구현 투자 판단

| 설계 후보 | 판단 | 구현 비용과 실제로 줄이는 비용 |
|---|---|---|
| 전체 게임 RAM을 일반 HOST_VISIBLE 원본으로 교체 | 현 구조에서는 착수 보류 | 커널과 NCE의 주소 배치부터 바꿔도 나중에 요청되는 두 번째 CPU view가 남는다. 최초 alias 때만 기존 section으로 옮기는 변형도 모든 CPU 사용자와 GPU 제출을 멈추고, 이미 공개한 **동일 VA**를 다시 확보해야 한다. 일반 드라이버 할당에서 이 인계가 보장되는 경로를 찾지 못했다. |
| Switch 2 그래픽 서버의 214 할당을 공용 원본으로 교체 | 214 전체에 적용할 근거 부족 | 최초 할당자를 알 수 있지만 일부 영역은 CPU가 상수를 쓰며, 216 요청은 게임이 자기 VA로 채운 메모리에 해당 타일을 다시 연결한다. 이미 lazy 처리하는 렌더 타깃은 실제 쓰기가 있을 때만 backing을 만든다. 예약한 MiB만으로 절감 가능한 복사량을 산정할 수 없다. 호스트 전용 소비·단일 view·수명 내 remap 부재를 계약으로 구별할 수 있고 그 범위의 비용이 기록될 때만 독립 후보가 된다. [214 생성 분기와 216 타일 재연결](../source/src/core/hle/service/nx2_gsv.h) |
| NVN compute에서 셰이더가 쓸 수 있는 범위만 되쓰기 대상으로 등록 | **먼저 구현할 가치 있음** | NCE 주소와 Vulkan allocation을 바꾸지 않고, 읽기 전용 SSBO나 넓게 잡힌 바인딩에 대해 생기는 불필요한 쓰기 등록, 제출 전후 사본과 병합 후보를 줄인다. 쓰기 자체가 필요한 범위에서는 현재의 일관성 처리가 유지된다. |
| 물리 backing 범위별 import/copy 혼합 | 조건부로 가치 있음 | CPU가 자주 쓰는 물리 범위와 GPU atomic 등으로 import가 비싼 범위가 분리될 때 유리하다. 실제 쓰기 범위를 바로잡기 전에 승격하면 읽기 전용 compute 입력 한 개로 256 MiB 단위가 copy로 고정될 수 있다. 같은 PA의 alias, 캐시된 descriptor와 진행 중 BDA 표의 전환까지 필요해 위 후보보다 범위가 크다. [기존 혼합 설계](GPU_MEMORY_HYBRID_DESIGN.md) |

### 먼저 구현할 후보를 고른 근거

- **오카리나의 원본 로그는 기존 혼합 설계서의 원인 설명과 다르다.** 192.35초의 같은 집계 창(120회 제출)에서 pre 723.0 MB, post 723.0 MB, raw write log 0개, 모든 청크를 표시한 raw 바인딩 누계 0회다. 시작 시 해당 게임의 raw pool table이 꺼져 있었다. 프레임 711은 전체 213.0 ms, GPU 45.9 ms였다. 이 실행의 사본량을 raw 바인딩 탓으로 돌릴 수 없다. 723 MB 중 각 쓰기 등록 경로의 몫이나 실제 절감 가능한 양은 이 로그가 분리해 주지 않는다. [오카리나 원본 로그](../logs_poolcopy/ab_copy_205455.txt)
- **현재 소스에는 쓰기 범위를 과대 등록하는 구체적인 경로가 있다.** 셰이더 변환기는 SSBO별 `is_written`과 store·atomic의 도달 상한 `write_extent`를 생성한다. Eden NVN 브리지는 전자만 전달한다. Windows 바인더는 compute(`st == 5`)이면 전달받은 읽기 전용 판정과 관계없이 `vk_gpu_wrote(a,n)`을 호출한다. 기존 Windows 바인더에도 유사한 분기가 있다. `sub_gpu_wrote → pc_pre_note → pc_pre_range`에서 동기화된 사본이 있으면 제출 전 복사가 기록된다. 256개 이상 세부 범위를 처리한 같은 청크는 전체 범위로 넓어질 수도 있다. [셰이더 분석](../source/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp) · [상한 계산](../source/src/shader_recompiler/storage_write_extent.h) · [NVN 브리지](../source/src/video_core/nvn/nvn_shader_bridge.cpp) · [Eden 바인더](../host/dm/windows/gfx_vk_eden.inc) · [기존 바인더](../host/dm/windows/gfx_vk_exec.c) · [복사 등록](../host/dm/windows/gfx_vk_core.c)
- **최소 설계.** 바인딩 가능한 길이와 dispatch가 수정할 수 있는 길이를 분리한다. 완전하게 분석된 읽기 전용 바인딩은 GPU 읽기 의존성과 수명을 추적하되 쓰기 등록을 생략한다. 쓰기 가능한 SSBO는 실제 dispatch 크기와 기존 `ResolveStorageWriteExtent`로 상한이 증명된 경우에만 추적 범위를 줄인다. 동적·간접 접근에서 상한을 못 구하거나, reflection이 누락됐거나, 별도 raw·alias 경로가 같은 바이트에 쓸 수 있으면 해당 쓰기는 보수적으로 추적한다. 같은 바이트의 다른 쓰기, CPU와 GPU의 충돌 병합, 실제 Vulkan descriptor 길이는 그대로 보존해야 한다. 오래된 shader cache에 새 정보가 없으면 범위 미상으로 다룬다.
- **젤다무쌍2에서는 다른 이득도 가능하다.** 150.24~151.63초의 다섯 집계에서 각각 764, 763, 761, 767, 762건이 CPU-only로 분류됐고 GPU-written은 모두 0이었다. 이 수치는 CPU가 쓰는 페이지에 사본 관리 부담이 걸렸다는 근거다. 같은 256 MiB 물리 범위의 *다른* 페이지에도 GPU 쓰기가 없었다는 뜻은 아니다. 따라서 범위 전체의 import 유지 또는 copy 승격 판단은 실제 물리 범위별 분포에 의존한다. [젤다무쌍2 원본 로그](../logs_poolcopy/ab_copy_204204.txt)

투자 판단을 뒤집을 기준도 있다. 쓰기 전후 사본 바이트의 대부분이 실제 GPU 쓰기의 증명 가능한 범위 안에 든다면 SSBO 분석으로 회수할 수 있는 비용은 작다. 반대로 전용 할당의 반복 import·복사 비용이 크고 그 수명에 CPU 별칭이 필요 없다는 근거가 나오면, 그 할당에 한정한 드라이버 공용 원본의 우선순위가 올라간다. 현재 자료로는 어느 후보도 게임 FPS 개선량을 확정할 수 없다.
