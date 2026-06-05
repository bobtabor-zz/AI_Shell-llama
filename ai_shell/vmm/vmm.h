#ifndef VMM_H
#define VMM_H

#include <stddef.h>
#include <stdint.h>

typedef struct vmm vmm_t;

// ----------------------
// Flags (restored)
// ----------------------
typedef uint32_t vmm_flags_t;

enum {
    VMM_FLAG_NONE       = 0u,
    VMM_FLAG_LARGEPAGES = 1u << 0,
    VMM_FLAG_LOCKED     = 1u << 1,
};

// ----------------------
// Region descriptor
// ----------------------
typedef struct vmm_region {
    void *      ptr;  // temporary mapping (optional)
    size_t      size;
    vmm_flags_t flags;
    int         numa_node;
    size_t      offset;  // absolute offset in vmm.bin
} vmm_region_t;

// ----------------------
// Config
// ----------------------
typedef struct vmm_config {
    size_t budget_bytes;
    int    prefer_largepages;
    int    prefetch_on_alloc;
    int    default_numa_node;
} vmm_config_t;

// ----------------------
// API
// ----------------------
vmm_t * vmm_init(const vmm_config_t * cfg);
void    vmm_shutdown(vmm_t * v);

typedef struct vmm vmm_t;

size_t vmm_budget(vmm_t * v);
size_t vmm_used(vmm_t * v);

int vmm_alloc(vmm_t * v, size_t bytes, vmm_flags_t flags, int numa_node, vmm_region_t * out);
int vmm_free(vmm_t * v, vmm_region_t * r);

void * vmm_map_window(vmm_t * v, size_t offset, size_t length);
void   vmm_unmap_window(vmm_t * v);

// simple engine-facing API
void   vmm_destroy(vmm_t* v);
void* vmm_map(vmm_t* v, uint64_t offset, size_t length);
void   vmm_unmap(vmm_t* v, void* ptr, size_t length);


#endif
