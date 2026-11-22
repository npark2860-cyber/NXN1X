#pragma once

#include <memory>
#include <cstdint>

namespace Common { class DynamicLibrary; }
namespace Vulkan::NvnStreamline {
std::shared_ptr<Common::DynamicLibrary> Open();
void Shutdown();
bool Active();
void Configure(void* physical_device, void* window);
void ResetHistory();
unsigned OpticalTarget(uint64_t ticket);
void BeforePresent(uint64_t ticket, const uint32_t* region);
void AfterPresent(uint64_t ticket, uint64_t original_ticket, bool accepted, bool generated);
}
