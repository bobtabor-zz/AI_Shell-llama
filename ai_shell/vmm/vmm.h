#ifndef VMM_H
#define VMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct vmm vmm_t;

#define VMM_MODEL_MAGIC 0x564D4D31u  // 'VMM1' or whatever you prefer

    typedef struct {
        uint32_t magic;        // VMM_MODEL_MAGIC
        uint32_t version;      // 1
        uint32_t tensor_count; // number of tensors
        uint64_t meta_offset;  // offset of meta table in vmm.bin
        uint64_t data_offset;  // offset of first tensor data in vmm.bin
    } vmm_model_header_t;

    typedef struct {
        char     name[128];
        uint32_t dtype;
        uint32_t n_dims;
        uint64_t ne[4];
        uint64_t size_bytes;
        uint64_t offset;       // offset in vmm.bin
    } vmm_tensor_meta_t;

    typedef struct {
        char     name[128];
        uint32_t dtype;
        uint32_t n_dims;
        uint64_t ne[4];
        uint64_t size_bytes;
        uint64_t file_offset;
    } gguf_tensor_desc_t;


    typedef struct vmm_model {
        vmm_t* vmm;
        uint32_t         tensor_count;
        uint64_t         meta_offset;
        uint64_t         data_offset;
        vmm_tensor_meta_t* meta;  // mapped meta table
    } vmm_model_t;


    typedef struct {
        size_t budget_bytes;
        int    prefer_largepages;
        int    prefetch_on_alloc;
        int    default_numa_node;
    } vmm_config_t;

    typedef uint32_t vmm_flags_t;

    typedef struct {
        uint64_t   offset;    // offset in vmm.bin
        size_t     size;      // bytes
        vmm_flags_t flags;
        int        numa_node;
        void* ptr;       // mapped pointer (if mapped)
    } vmm_region_t;

    typedef struct vmm_model vmm_model_t;

    vmm_model_t* vmm_model_open(
        const char* gguf_path,
        const char* vmm_path,
        size_t      vmm_budget_bytes);
    

    void vmm_model_close(vmm_model_t* m);


    // create/open modes
    typedef enum {
        VMM_MODE_CREATE_TRUNCATE = 0,  // build from GGUF, overwrite vmm.bin
        VMM_MODE_OPEN_EXISTING = 1   // use existing vmm.bin, do NOT truncate
    } vmm_mode_t;

    // core lifecycle
    vmm_t* vmm_create(const char* path, size_t size_bytes, vmm_mode_t mode);
    void    vmm_destroy(vmm_t* v);

    // stats
    size_t  vmm_budget(vmm_t* v);
    size_t  vmm_used(vmm_t* v);

    // alloc
    int     vmm_alloc(vmm_t* v, size_t bytes, vmm_flags_t flags, int numa, vmm_region_t* out);
    int     vmm_free(vmm_t* v, vmm_region_t* r);

    // mapping
    void* vmm_map(vmm_t* v, uint64_t offset, size_t length);
    void    vmm_unmap(vmm_t* v, void* ptr, size_t length);

    // pin/prefetch
    int     vmm_pin(vmm_t* v, vmm_region_t* r);
    int     vmm_unpin(vmm_t* v, vmm_region_t* r);
    int     vmm_prefetch(vmm_t* v, void* ptr, size_t bytes);

    // NUMA
    void    vmm_set_default_numa(vmm_t* v, int node);
    int     vmm_get_default_numa(vmm_t* v);

#ifdef __cplusplus
}
#endif

#endif // VMM_H
