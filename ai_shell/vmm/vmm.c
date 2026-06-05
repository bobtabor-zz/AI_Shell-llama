#include "vmm.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

typedef struct vmm_state {
    HANDLE    file;
    HANDLE    mapping;
    uint64_t  size;
    uint64_t  used;
    int       default_numa_node;
    uint8_t * window_base;
    uint64_t  window_offset;
    uint64_t  window_size;
} vmm_state_t;

static vmm_state_t g_vmm;

struct vmm {
    vmm_config_t cfg;
    size_t       used;
};

// ----------------------
// Init
// ----------------------
static int vmm_disk_init(const char * path, uint64_t size_bytes) {
    LARGE_INTEGER li;
    li.QuadPart = size_bytes;

    g_vmm.file = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                             CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_RANDOM_ACCESS, NULL);
    if (g_vmm.file == INVALID_HANDLE_VALUE) {
        return -1;
    }

    if (!SetFilePointerEx(g_vmm.file, li, NULL, FILE_BEGIN) || !SetEndOfFile(g_vmm.file)) {
        return -2;
    }

    g_vmm.mapping = CreateFileMappingA(g_vmm.file, NULL, PAGE_READWRITE, li.HighPart, li.LowPart, NULL);
    if (!g_vmm.mapping) {
        return -3;
    }

    // NEW: map the entire file into memory
    g_vmm.window_base = MapViewOfFile(g_vmm.mapping, FILE_MAP_ALL_ACCESS,
                                      0,          // high offset
                                      0,          // low offset
                                      size_bytes  // bytes to map (0 = whole file, but be explicit)
    );
    if (!g_vmm.window_base) {
        return -4;
    }

    g_vmm.size          = size_bytes;
    g_vmm.used          = 0;
    g_vmm.window_offset = 0;
    g_vmm.window_size   = size_bytes;

    return 0;
}

vmm_t * vmm_init(const vmm_config_t * cfg) {
    if (!cfg) {
        return NULL;
    }

    if (vmm_disk_init("vmm.bin", cfg->budget_bytes) != 0) {
        return NULL;
    }

    // keep whatever you already do with g_vmm
    g_vmm.size              = cfg->budget_bytes;  // ← add this line if it’s not there
    g_vmm.used              = 0;
    g_vmm.default_numa_node = cfg->default_numa_node;

    vmm_t * v = (vmm_t *) calloc(1, sizeof(vmm_t));
    if (!v) {
        return NULL;
    }

    v->cfg  = *cfg;  // if you already have this, keep it
    v->used = 0;

    return v;
}

void vmm_shutdown(vmm_t * v) {
    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
    }

    if (g_vmm.mapping) {
        CloseHandle(g_vmm.mapping);
    }

    if (g_vmm.file) {
        CloseHandle(g_vmm.file);
    }

    if (v) {
        free(v);
    }
}


size_t vmm_budget(vmm_t * v) {
    (void) v;           // we don’t trust or need the handle here
    return g_vmm.size;  // single source of truth for the budget
}

size_t vmm_used(vmm_t * v) {
    (void) v;  // ignore the handle
    return g_vmm.used;
}

// ----------------------
// Alloc
// ----------------------
int vmm_alloc(vmm_t * v, size_t bytes, vmm_flags_t flags, int numa, vmm_region_t * out) {
    (void) v;  // handle is ignored

    uint64_t align   = 64ull * 1024ull;
    uint64_t current = g_vmm.used;
    uint64_t aligned = (current + align - 1) & ~(align - 1);

    if (aligned + bytes > g_vmm.size) {
        return -1;
    }

    // ensure the requested range fits in the current window
    if (!g_vmm.window_base || aligned + bytes > g_vmm.window_size) {
        // you can either fail, or later implement window remapping here
        return -3;
    }

    out->offset    = aligned;
    out->size      = bytes;
    out->flags     = flags;
    out->numa_node = numa;

    // real RAM pointer into the mapped VMM.bin window
    out->ptr = (uint8_t *) g_vmm.window_base + aligned;

    g_vmm.used = aligned + bytes;

    return 0;
}

// ----------------------
// Window map
// ----------------------
void * vmm_map_window(vmm_t * v, size_t offset, size_t length) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    size_t gran = si.dwAllocationGranularity;

    size_t aligned = offset & ~(gran - 1);
    size_t delta   = offset - aligned;
    size_t map_sz  = length + delta;

    if (g_vmm.window_base && aligned >= g_vmm.window_offset &&
        aligned + map_sz <= g_vmm.window_offset + g_vmm.window_size) {
        return g_vmm.window_base + (aligned - g_vmm.window_offset) + delta;
    }

    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
    }

    uint32_t lo = (uint32_t) (aligned & 0xFFFFFFFFull);
    uint32_t hi = (uint32_t) (aligned >> 32);

    g_vmm.window_base = (uint8_t *) MapViewOfFile(g_vmm.mapping, FILE_MAP_ALL_ACCESS, hi, lo, map_sz);
    if (!g_vmm.window_base) {
        return NULL;
    }

    g_vmm.window_offset = aligned;
    g_vmm.window_size   = map_sz;

    return g_vmm.window_base + delta;
}

void vmm_unmap_window(vmm_t * v) {
    if (g_vmm.window_base) {
        UnmapViewOfFile(g_vmm.window_base);
        g_vmm.window_base   = NULL;
        g_vmm.window_offset = 0;
        g_vmm.window_size   = 0;
    }
}

void vmm_set_default_numa(vmm_t * v, int node) {
    if (!v) {
        return;
    }
    v->cfg.default_numa_node = node;
}

int vmm_get_default_numa(vmm_t * v) {
    (void) v;  // handle is just a token now
    return g_vmm.default_numa_node;
}

vmm_t * vmm_create(const char * path, size_t size_bytes) {
    (void) path;  // unused for now

    vmm_config_t cfg;
    cfg.budget_bytes      = size_bytes;
    cfg.prefer_largepages = 0;
    cfg.prefetch_on_alloc = 0;
    cfg.default_numa_node = -1;

    return vmm_init(&cfg);
}

int vmm_free(vmm_t * v, vmm_region_t * r) {
    if (!v || !r || !r->ptr) {
        return -1;
    }
    // bump allocator: no per‑region free
    memset(r, 0, sizeof(*r));
    return 0;
}

int vmm_pin(vmm_t * v, vmm_region_t * r) {
#ifdef _WIN32
    if (!v || !r || !r->ptr) {
        return -1;
    }
    if (VirtualLock(r->ptr, r->size)) {
        return 0;
    }
    return -1;
#else
    (void) v;
    (void) r;
    return -99;
#endif
}

int vmm_unpin(vmm_t * v, vmm_region_t * r) {
#ifdef _WIN32
    if (!v || !r || !r->ptr) {
        return 0;
    }
    if (VirtualUnlock(r->ptr, r->size)) {
        return 0;
    }
    return -1;
#else
    (void) v;
    (void) r;
    return -99;
#endif
}

int vmm_prefetch(vmm_t * v, void * ptr, size_t bytes) {
#ifdef _WIN32
    //typedef struct _WIN32_MEMORY_RANGE_ENTRY { PVOID VirtualAddress; SIZE_T NumberOfBytes; } WIN32_MEMORY_RANGE_ENTRY;
    //WIN32_MEMORY_RANGE_ENTRY range = { ptr, (SIZE_T)bytes };
    //PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);

    WIN32_MEMORY_RANGE_ENTRY range = { ptr, (SIZE_T) bytes };
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);

    return 0;
#else
    (void) v;
    (void) ptr;
    (void) bytes;
    return -99;
#endif
}

// -----------------------------------------------------------------------------
// Engine-facing wrappers
// -----------------------------------------------------------------------------

void vmm_destroy(vmm_t* v) {
    // reuse existing shutdown logic
    vmm_shutdown(v);
}

void* vmm_map(vmm_t* v, uint64_t offset, size_t length) {
    // engine uses 64-bit offsets; window API uses size_t
    (void)v;  // handle is not needed by current implementation
    return vmm_map_window(v, (size_t)offset, length);
}

void vmm_unmap(vmm_t* v, void* ptr, size_t length) {
    (void)v;
    (void)ptr;
    (void)length;
    // current design has a single window; just drop it
    vmm_unmap_window(v);
}
