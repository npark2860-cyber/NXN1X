// Production archive stream with an in-process Vulkan double. No loader/device.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static int short_write, flush_fail;
static size_t test_write(const void *p,size_t s,size_t n,FILE *f) { return short_write && n ? n-1 : fwrite(p,s,n,f); }
static int test_flush(FILE *f) { return flush_fail?EOF:fflush(f); }
#define fwrite test_write
#define fflush test_flush
#include "../nvn_image_archive.h"
#undef fwrite
#undef fflush
static unsigned checks, submits, barriers, invalidates, upload_checks;
static int queue_locked, allocation_fail, wait_fail, mutate_after_wait;
static uint64_t generation=1;
static uint8_t mapped[NVNA_STAGING_BYTES];
static nvna_shape shape;
#define CHECK(x) do {++checks;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#define HANDLE(t) ((t)(uintptr_t)1)
static unsigned char pattern(unsigned aspect,unsigned lv,unsigned layer,unsigned z,unsigned row,unsigned byte) {
    return (unsigned char)(aspect*17+lv*13+layer*7+z*3+row*11+byte);
}
static void copy_regions(uint32_t n,const VkBufferImageCopy *copies,int upload) {
    for(uint32_t i=0;i<n;i++) {
        const VkBufferImageCopy *c=&copies[i];
        const unsigned aspect=c->imageSubresource.aspectMask;
        unsigned row_bytes=(c->imageExtent.width+shape.block_width-1)/shape.block_width*nvna_aspect_bytes(&shape,aspect);
        unsigned rows=(c->imageExtent.height+shape.block_height-1)/shape.block_height;
        for(unsigned y=0;y<rows;y++) for(unsigned x=0;x<row_bytes;x++) {
            unsigned char v=pattern(aspect,c->imageSubresource.mipLevel,c->imageSubresource.baseArrayLayer,
                c->imageOffset.z,(unsigned)c->imageOffset.y/shape.block_height+y,x);
            uint8_t *dst=mapped+c->bufferOffset+(size_t)y*row_bytes+x;
            if(upload) { if(*dst!=v) {fprintf(stderr,"transfer mismatch\n");exit(1);} } else *dst=v;
        }
    }
    if(upload) upload_checks++;
}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(VkDevice d,const VkBufferCreateInfo *i,const VkAllocationCallbacks *a,VkBuffer *b) {(void)d;(void)i;(void)a;*b=HANDLE(VkBuffer);return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(VkDevice d,VkBuffer b,VkMemoryRequirements *r){(void)d;(void)b;*r=(VkMemoryRequirements){NVNA_STAGING_BYTES,256,1};}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(VkDevice d,const VkMemoryAllocateInfo *i,const VkAllocationCallbacks *a,VkDeviceMemory *m){(void)d;(void)i;(void)a;if(allocation_fail)return VK_ERROR_OUT_OF_DEVICE_MEMORY;*m=HANDLE(VkDeviceMemory);return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(VkDevice d,VkBuffer b,VkDeviceMemory m,VkDeviceSize o){(void)d;(void)b;(void)m;(void)o;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize s,VkMemoryMapFlags f,void **p){(void)d;(void)m;(void)o;(void)s;(void)f;*p=mapped;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(VkDevice d,const VkCommandPoolCreateInfo *i,const VkAllocationCallbacks *a,VkCommandPool *p){(void)d;(void)i;(void)a;*p=HANDLE(VkCommandPool);return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(VkDevice d,const VkCommandBufferAllocateInfo *i,VkCommandBuffer *b){(void)d;(void)i;*b=HANDLE(VkCommandBuffer);return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(VkDevice d,const VkFenceCreateInfo *i,const VkAllocationCallbacks *a,VkFence *f){(void)d;(void)i;(void)a;*f=HANDLE(VkFence);return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(VkDevice d,uint32_t n,const VkFence *f,VkBool32 all,uint64_t timeout){(void)d;(void)n;(void)f;(void)all;(void)timeout;CHECK(!queue_locked);if(wait_fail)return VK_ERROR_OUT_OF_HOST_MEMORY;if(mutate_after_wait)generation++;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice d,VkCommandPool p,VkCommandPoolResetFlags f){(void)d;(void)p;(void)f;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer b,const VkCommandBufferBeginInfo *i){(void)b;(void)i;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer b){(void)b;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice d,uint32_t n,const VkFence *f){(void)d;(void)n;(void)f;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(VkQueue q,uint32_t n,const VkSubmitInfo *s,VkFence f){(void)q;(void)n;(void)s;(void)f;CHECK(queue_locked);submits++;return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(VkDevice d,uint32_t n,const VkMappedMemoryRange *r){(void)d;(void)n;CHECK(r->offset==0);return VK_SUCCESS;}
VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice d,uint32_t n,const VkMappedMemoryRange *r){(void)d;(void)n;CHECK(r->size==VK_WHOLE_SIZE || !(r->size%256));invalidates++;return VK_SUCCESS;}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(VkCommandBuffer c,VkImage i,VkImageLayout l,VkBuffer b,uint32_t n,const VkBufferImageCopy *r){(void)c;(void)i;(void)l;(void)b;copy_regions(n,r,0);}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(VkCommandBuffer c,VkBuffer b,VkImage i,VkImageLayout l,uint32_t n,const VkBufferImageCopy *r){(void)c;(void)i;(void)l;(void)b;copy_regions(n,r,1);}
VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(VkCommandBuffer c,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags f,uint32_t nm,const VkMemoryBarrier *m,uint32_t nb,const VkBufferMemoryBarrier *b,uint32_t ni,const VkImageMemoryBarrier *i){(void)c;(void)f;(void)nm;(void)m;(void)nb;(void)b;if(ni && s==VK_PIPELINE_STAGE_TRANSFER_BIT && d==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && i->srcAccessMask==VK_ACCESS_TRANSFER_READ_BIT)barriers++;}
VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice d,VkFence f,const VkAllocationCallbacks *a){(void)d;(void)f;(void)a;}
VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice d,VkCommandPool p,const VkAllocationCallbacks *a){(void)d;(void)p;(void)a;}
VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice d,VkDeviceMemory m){(void)d;(void)m;}
VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice d,VkBuffer b,const VkAllocationCallbacks *a){(void)d;(void)b;(void)a;}
VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice d,VkDeviceMemory m,const VkAllocationCallbacks *a){(void)d;(void)m;(void)a;}
static void lock(void){CHECK(!queue_locked);queue_locked=1;}
static void unlock(void){CHECK(queue_locked);queue_locked=0;}
static int changed(void *p){return generation!=*(uint64_t *)p;}
static VkPhysicalDeviceMemoryProperties mem={.memoryTypeCount=1,.memoryTypes={{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,0}}};
static VkResult transfer(nvna_workspace *w,FILE *f,int up) {
    return nvna_stream_cached(w,HANDLE(VkDevice),HANDLE(VkQueue),0,&mem,HANDLE(VkImage),&shape,f,up,lock,unlock,0);
}
int main(void) {
    nvna_workspace w={.non_coherent_atom=256}; FILE *f=tmpfile(); CHECK(f);
    shape=(nvna_shape){VK_FORMAT_R8_UNORM,VK_IMAGE_ASPECT_COLOR_BIT,2048,2048,1,2,3,1,1,1};
    CHECK(transfer(&w,f,0)==VK_SUCCESS); CHECK(submits>1 && invalidates>1 && !barriers);
    CHECK((uint64_t)ftell(f)==nvna_size(&shape)); CHECK(transfer(&w,f,1)==VK_SUCCESS && upload_checks>1);
    uint64_t version=generation; w.cancelled=changed; w.cancel_context=&version;
    CHECK(transfer(&w,f,0)==VK_SUCCESS && barriers>0); CHECK(!w.pending);
    unsigned before=submits; mutate_after_wait=1;
    CHECK(transfer(&w,f,0)==VK_NOT_READY && submits==before+1 && !w.pending);
    mutate_after_wait=0; version=generation;
    short_write=1; CHECK(transfer(&w,f,0)==VK_ERROR_UNKNOWN && !w.pending); short_write=0;
    flush_fail=1; CHECK(transfer(&w,f,0)==VK_ERROR_UNKNOWN && !w.pending); flush_fail=0;
    wait_fail=1; CHECK(transfer(&w,f,0)==VK_ERROR_OUT_OF_HOST_MEMORY && w.pending);
    wait_fail=0; CHECK(nvna_workspace_wait(&w)==VK_SUCCESS && !w.pending);
    // Odd compressed dimensions and separate depth/stencil planes round trip.
    shape=(nvna_shape){VK_FORMAT_BC3_UNORM_BLOCK,VK_IMAGE_ASPECT_COLOR_BIT,19,23,2,3,5,4,4,16};
    CHECK(transfer(&w,f,0)==VK_SUCCESS && transfer(&w,f,1)==VK_SUCCESS);
    shape=(nvna_shape){VK_FORMAT_D32_SFLOAT_S8_UINT,VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT,61,37,1,3,4,1,1,8};
    CHECK(transfer(&w,f,0)==VK_SUCCESS && transfer(&w,f,1)==VK_SUCCESS);
    CHECK(nvna_workspace_destroy(&w)==VK_SUCCESS); fclose(f);
    allocation_fail=1;
    CHECK(nvna_workspace_init(&w,HANDLE(VkDevice),0,&mem,NVNA_STAGING_BYTES)==VK_ERROR_OUT_OF_DEVICE_MEMORY);
    CHECK(!w.pending && !w.buffer && !w.memory);
    printf("PASS production archive CPU double %u checks; full subresources, cancellation, short writes, failed waits, noncoherent visibility; real GPU NOT_RUN\n",checks);
}
