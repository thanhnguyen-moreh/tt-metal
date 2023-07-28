#include "llrt.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "hostdevcommon/common_values.hpp"

#include "build_kernels_for_riscv/build_kernel_options.hpp"

#include <unordered_set>
#include <mutex>

#include "tools/cpuprof/cpuprof.h"

namespace tt {

// llrt = lower-level runtime
namespace llrt {

namespace fs = std::filesystem;

using std::endl;
using std::move;
using std::string;
using std::to_string;
using std::uint32_t;
using std::unordered_map;
using std::vector;

struct HexNameToMemVectorCache {
    using lock = std::unique_lock<std::mutex>;
    // maps from RisckCacheMapKey to hex file path
    static HexNameToMemVectorCache &inst() {
        static HexNameToMemVectorCache inst_;
        return inst_;
    }

    bool exists(const string &path) {
        lock l(mutex_);
        return cache_.find(path) != cache_.end();
    }
    ll_api::memory &get(const string &path) {
        lock l(mutex_);
        return cache_[path];
    }
    void add(const string &path, ll_api::memory &mem) {
        lock l(mutex_);
        cache_[path] = mem;
    }

    unordered_map<string, ll_api::memory> cache_;
    std::mutex mutex_;
};

// made these free functions -- they're copy/paste of the member functions
// TODO: clean-up epoch_loader / epoch_binary -- a bunch of functions there should not be member functions
ll_api::memory get_risc_binary(string path, int chip_id, bool fw_build) {

    string path_to_bin = (fw_build ? get_firmware_compile_outpath(chip_id) : get_kernel_compile_outpath(chip_id)) + path;
    if (HexNameToMemVectorCache::inst().exists(path)) {
        // std::cout << "-- HEX2MEM CACHE HIT FOR " << path << std::endl;
        return HexNameToMemVectorCache::inst().get(path);
    }

    fs::path bin_file(path_to_bin);
    if (!fs::exists(bin_file)) {
        string tt_metal_home = string(getenv("TT_METAL_HOME"));
        // try loading from home in case cwd isn't home
        path_to_bin = tt_metal_home + "/" + path_to_bin;
        fs::path bin_file_h(path_to_bin);
        if (!fs::exists(bin_file_h)) {
            std::cout << " Error: " << bin_file.c_str() << " doesn't exist" << endl;
            TT_ASSERT(false);
        }
    }

    std::ifstream hex_istream(path_to_bin);
    ll_api::memory mem(hex_istream);

    // add this path to binary cache
    HexNameToMemVectorCache::inst().add(path, mem);

    return mem;
}

// This deasserts reset for all BRISCs (on all devices, all cores), but not other RISC processors (NCRISC, TRISC)
// Every core gets valid FW (blank kernel if nothing is running on the core) before being taken out ot reset
// This avoids the issue of cores running garbahe out of their L1
// TODO: deassert reset only for used BRISCs (needs a new deassert function w/ a list of core to de-assert)
void deassert_brisc_reset_for_all_chips_all_cores(tt_cluster *cluster, bool stagger_start) {
    cluster->deassert_risc_reset(stagger_start);
    log_debug(tt::LogLLRuntime, "deasserted reset for all BRISCs");
}

// TODO: try using "stop" method from device instead, it's the proper way of asserting reset
void assert_reset_for_all_chips(tt_cluster *cluster) {
    TT_ASSERT((cluster->type == tt::TargetDevice::Silicon) or (cluster->type == tt::TargetDevice::Versim));

    if (cluster->type == tt::TargetDevice::Silicon) {
        log_debug(tt::LogLLRuntime, "Starting resets for {} chips", cluster->get_num_chips());
        for (const chip_id_t &chip_id : cluster->get_all_chips()) {
            cluster->broadcast_remote_tensix_risc_reset(chip_id, TENSIX_ASSERT_SOFT_RESET);
        }
    }
}

// CoreCoord core --> NOC coordinates ("functional workers" from the SOC descriptor)
// NOC coord is also synonymous to routing / physical coord
// dram_channel id (0..7) for GS is also mapped to NOC coords in the SOC descriptor

void write_hex_vec_to_core(
    tt_cluster *cluster, int chip, const CoreCoord &core, std::vector<uint32_t> hex_vec, uint64_t addr, bool small_access) {
    // the API is named "write_dram_vec", and its overloaded variant is taking (chip, core) pair, ie. it can write to
    // core's L1
    cluster->write_dram_vec(hex_vec, tt_cxy_pair(chip, core), addr, small_access);
}

std::vector<std::uint32_t> read_hex_vec_from_core(
    tt_cluster *cluster, int chip, const CoreCoord &core, uint64_t addr, uint32_t size) {
    vector<std::uint32_t> read_hex_vec;
    cluster->read_dram_vec(read_hex_vec, tt_cxy_pair(chip, core), addr, size);
    return read_hex_vec;
}

void print_worker_cores(tt_cluster *cluster, chip_id_t chip_id) {
    std::cout << std::endl << "worker cores: " << std::endl;
    for (const CoreCoord &core : cluster->get_soc_desc(chip_id).workers) {
        std::cout << core.str() << " ";
    }
    std::cout << std::endl << std::endl;
}

bool is_worker_core(tt_cluster *cluster, const CoreCoord &core, chip_id_t chip_id) {
    return std::find(
               cluster->get_soc_desc(chip_id).workers.begin(), cluster->get_soc_desc(chip_id).workers.end(), core) !=
           cluster->get_soc_desc(chip_id).workers.end();
}

CircularBufferConfigVec create_circular_buffer_config_vector() {
    CircularBufferConfigVec circular_buffer_config_vec(
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG, 0);  // init to 0's
    return circular_buffer_config_vec;
}

void set_config_for_circular_buffer(
    CircularBufferConfigVec &circular_buffer_config_vec,
    uint32_t circular_buffer_index,
    uint32_t addr_in_bytes,
    uint32_t size_in_bytes,
    uint32_t num_pages) {

    uint32_t page_size = size_in_bytes / num_pages;
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index) =
        addr_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 1) =
        size_in_bytes >> 4;  // convert to addr in 16B words
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 2) = num_pages;
    circular_buffer_config_vec.at(UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * circular_buffer_index + 3) = page_size >> 4;
}

void write_circular_buffer_config_vector_to_core(
    tt_cluster *cluster, int chip, const CoreCoord &core, CircularBufferConfigVec circular_buffer_config_vec) {
    write_hex_vec_to_core(cluster, chip, core, circular_buffer_config_vec, CIRCULAR_BUFFER_CONFIG_BASE);
}

void write_graph_interpreter_op_info_to_core(
    tt_cluster *cluster, int chip, const CoreCoord &core, op_info_t op_info, int op_idx) {
    vector<uint32_t> op_info_vec = {
        op_info.op_code,
        op_info.cb_in0_id,
        op_info.cb_in1_id,
        op_info.cb_out_id,
        op_info.pop0,
        op_info.pop1,
        op_info.unary};
    uint32_t offset = op_info_vec.size() * sizeof(uint32_t) * op_idx;

    write_hex_vec_to_core(cluster, chip, core, op_info_vec, OP_INFO_BASE_ADDR + offset);
}

ll_api::memory read_mem_from_core(
    tt_cluster *cluster, int chip, const CoreCoord &core, const ll_api::memory& mem, uint64_t local_init_addr) {

    ll_api::memory read_mem;
    read_mem.fill_from_mem_template(mem, [&](std::vector<uint32_t>::iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = relocate_dev_addr(addr, local_init_addr);
        cluster->read_dram_vec(&*mem_ptr, tt_cxy_pair(chip, core), relo_addr, len * sizeof(uint32_t));
    });
    return read_mem;
}

void program_brisc_startup_addr(tt_cluster* cluster, int chip_id, const CoreCoord &core) {
    // Options for handling brisc fw not starting at mem[0]:
    // 1) Program the register for the start address out of reset
    // 2) Encode a jump in crt0 for mem[0]
    // 3) Write the jump to mem[0] here
    // This does #3.  #1 may be best, #2 gets messy (elf files
    // drop any section before .init, crt0 needs ifdefs, etc)
    vector<uint32_t> jump_to_fw;
    constexpr uint32_t jal_opcode = 0x6f;
    constexpr uint32_t jal_max_offset = 0x0007ffff;
    uint32_t opcode = jal_opcode;
    assert(MEM_BRISC_FIRMWARE_BASE < jal_max_offset);
    // See riscv spec for offset encoding below
    uint32_t jal_offset_bit_20 = 0;
    uint32_t jal_offset_bits_10_to_1 = (MEM_BRISC_FIRMWARE_BASE & 0x7fe) << 20;
    uint32_t jal_offset_bit_11 = (MEM_BRISC_FIRMWARE_BASE & 0x800) << 9;
    uint32_t jal_offset_bits_19_to_12 = (MEM_BRISC_FIRMWARE_BASE & 0xff000) << 0;
    uint32_t jal_offset =
        jal_offset_bit_20 |
        jal_offset_bits_10_to_1 |
        jal_offset_bit_11 |
        jal_offset_bits_19_to_12;
    jump_to_fw.push_back(jal_offset | opcode);
    write_hex_vec_to_core(cluster, chip_id, core, jump_to_fw, 0);
}

static bool test_load_write_read_risc_binary_imp(
    tt_cluster *cluster, ll_api::memory &mem, int chip_id, const CoreCoord &core, int riscv_id) {

    assert(is_worker_core(cluster, core, chip_id));

    uint64_t local_init_addr;
    switch (riscv_id) {
        case 0: local_init_addr = MEM_BRISC_INIT_LOCAL_L1_BASE; break;
        case 1: local_init_addr = MEM_NCRISC_INIT_LOCAL_L1_BASE; break;
        case 2: local_init_addr = MEM_TRISC0_INIT_LOCAL_L1_BASE; break;
        case 3: local_init_addr = MEM_TRISC1_INIT_LOCAL_L1_BASE; break;
        case 4: local_init_addr = MEM_TRISC2_INIT_LOCAL_L1_BASE; break;
    }

    log_debug(tt::LogLLRuntime, "hex_vec size = {}, size_in_bytes = {}", mem.size(), mem.size()*sizeof(uint32_t));
    mem.process_spans([&](std::vector<uint32_t>::const_iterator mem_ptr, uint64_t addr, uint32_t len) {
        uint64_t relo_addr = relocate_dev_addr(addr, local_init_addr);

        cluster->write_dram_vec(&*mem_ptr, len, tt_cxy_pair(chip_id, core), relo_addr);
    });

    log_debug(tt::LogLLRuntime, "wrote hex to core {}", core.str().c_str());

    if (std::getenv("TT_KERNEL_READBACK_ENABLE") != nullptr) {
        ll_api::memory read_mem = read_mem_from_core(cluster, chip_id, core, mem, local_init_addr);
        log_debug(tt::LogLLRuntime, "read hex back from the core");
        return mem == read_mem;
    }

    return true;
}

bool test_load_write_read_risc_binary(
    tt_cluster *cluster, ll_api::memory &mem, int chip_id, const CoreCoord &core, int riscv_id) {

    test_load_write_read_risc_binary_imp(cluster, mem, chip_id, core, riscv_id);

    return true;
}

bool test_load_write_read_risc_binary(
    tt_cluster *cluster, std::string hex_file_name, int chip_id, const CoreCoord &core, int riscv_id, bool fw_build) {

    log_debug(tt::LogLLRuntime, "hex_file_path = {}", (fw_build ? get_firmware_compile_outpath(chip_id) : get_kernel_compile_outpath(chip_id)) + hex_file_name);
    ll_api::memory mem = get_risc_binary(hex_file_name, chip_id, fw_build);
    test_load_write_read_risc_binary_imp(cluster, mem, chip_id, core, riscv_id);

    return true;
}

// for TRISCs
bool test_load_write_read_trisc_binary(
    tt_cluster *cluster, std::string hex_file_name, int chip_id, const CoreCoord &core, int triscv_id) {

    assert(triscv_id >= 0 and triscv_id <= 2);
    return test_load_write_read_risc_binary(cluster, hex_file_name, chip_id, core, triscv_id + 2);
}

bool test_load_write_read_trisc_binary(
    tt_cluster *cluster, ll_api::memory &mem, int chip_id, const CoreCoord &core, int triscv_id) {

    assert(triscv_id >= 0 and triscv_id <= 2);
    return test_load_write_read_risc_binary(cluster, mem, chip_id, core, triscv_id + 2);
}

void disable_ncrisc(tt_cluster *cluster, int chip_id, const CoreCoord &core) {
    // disable NCRISC
    uint64_t use_ncrisc_addr = MEM_ENABLE_NCRISC_MAILBOX_ADDRESS;
    write_hex_vec_to_core(cluster, chip_id, core, {0}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "disabled ncrisc");
}

void enable_ncrisc(tt_cluster *cluster, int chip_id, const CoreCoord &core) {
    uint64_t use_ncrisc_addr = MEM_ENABLE_NCRISC_MAILBOX_ADDRESS;
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_ncrisc_addr);
    log_debug(tt::LogLLRuntime, "enabled ncrisc");
}

void enable_triscs(tt_cluster *cluster, int chip_id, const CoreCoord &core) {
    uint64_t use_triscs_addr = MEM_ENABLE_TRISC_MAILBOX_ADDRESS;
    write_hex_vec_to_core(cluster, chip_id, core, {1}, use_triscs_addr);
    log_debug(tt::LogLLRuntime, "enabled triscs");
}

void disable_triscs(tt_cluster *cluster, int chip_id, const CoreCoord &core) {
    uint64_t use_triscs_addr = MEM_ENABLE_TRISC_MAILBOX_ADDRESS;
    write_hex_vec_to_core(cluster, chip_id, core, {0}, use_triscs_addr);
    log_debug(tt::LogLLRuntime, "disabled triscs");
}

WorkerCores get_worker_cores_from_cluster(tt_cluster *cluster, int chip_id) {
    WorkerCores worker_cores;
    for (CoreCoord raw_core : cluster->get_soc_desc(chip_id).workers) {
        TT_ASSERT(cluster->get_soc_desc(chip_id).is_worker_core(raw_core));
        worker_cores.emplace_back(chip_id, raw_core);
    }
    return worker_cores;
}

CoreCoord get_core_for_dram_channel(tt_cluster *cluster, int dram_channel_id, chip_id_t chip_id) {
    return cluster->get_soc_desc(chip_id).get_preferred_worker_core_for_dram_channel(dram_channel_id);
}

namespace utils {
void log_current_ai_clk(tt_cluster *cluster) {
    if (cluster->type == tt::TargetDevice::Silicon) {
        for (const chip_id_t &chip_id : cluster->get_all_chips()) {
            int ai_clk = cluster->get_device_aiclk(chip_id);
            log_info(tt::LogLLRuntime, "AI CLK for device {} is:   {} MHz", chip_id, ai_clk);
        }
    }
}
}  // namespace utils

namespace internal_ {
// This loads to briscs and ncriscs - we may want to add TensixRiscsOptions here
void load_blank_kernel_to_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::vector<CoreCoord> cores) {
    TT_ASSERT(riscs_to_load != TensixRiscsOptions::NONE, "You must specify a non-NONE RISC to load blank kernels to");

    for (const CoreCoord &core : cores) {
        bool pass = true;

        // PROF_BEGIN("write_brisc")
        pass = test_load_write_read_risc_binary(cluster, "blank_op/brisc/brisc.hex", chip_id, core, 0);
        if (!pass) {
            throw std::runtime_error("Initial testing read/write of brisc to core failed");
        }  // PROF_END("write_brisc")

        if (deduce_if_involves_ncrisc(riscs_to_load)) {  // PROF_BEGIN("ncrisc")
            pass =
                test_load_write_read_risc_binary(cluster, "blank_op/ncrisc/ncrisc.hex", chip_id, core, 1);
            if (!pass) {
                throw std::runtime_error("Initial testing read/write of ncrisc to core failed");
            }
        }  // PROF_END("ncrisc")

        if (deduce_if_involves_triscs(riscs_to_load)) {  // PROF_BEGIN("trisc")
            string op_path = "blank_op";
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread0/tensix_thread0.hex", chip_id, core, 0);
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread1/tensix_thread1.hex", chip_id, core, 1);
            pass &= test_load_write_read_trisc_binary(
                cluster, op_path + "/tensix_thread2/tensix_thread2.hex", chip_id, core, 2);
            if (!pass) {
                throw std::runtime_error("Initial testing read/write of blank to trisc to core failed");
            }
        }  // PROF_END("trisc")
    }
}

void load_blank_kernel_to_all_worker_cores_with_exceptions(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions &riscs_to_load, std::unordered_set<CoreCoord> exceptions) {
    std::vector<CoreCoord> cores_to_load_with_blanks;  // PROF_BEGIN("set_diff")

    uint32_t harvested_noc_rows = 0;
    if (cluster->type == tt::TargetDevice::Silicon) {
        harvested_noc_rows = cluster->get_harvested_rows(chip_id);
    }
    for (const CoreCoord &worker_core : cluster->get_soc_desc(chip_id).workers) {
        unsigned int row = worker_core.y;
        bool row_harvested = (harvested_noc_rows>>row)&0x1;
        if (not row_harvested and exceptions.find(worker_core) == exceptions.end()) {
            cores_to_load_with_blanks.push_back(worker_core);
        }
    }
    // PROF_END("set_diff")

    for (const CoreCoord &core : cores_to_load_with_blanks) {  // PROF_BEGIN("log_blank")
        log_debug(tt::LogLLRuntime, "loading blank to core - {}", core.str());
    }  // PROF_END("log_blank")

    load_blank_kernel_to_cores(cluster, chip_id, riscs_to_load, cores_to_load_with_blanks);
}

void setup_riscs_on_specified_core(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const CoreCoord &core) {
    if (riscs_options == TensixRiscsOptions::NONE) {
        TT_THROW("You can't run nothing on the riscs on core " + core.str());
    }

    bool involves_triscs = deduce_if_involves_triscs(riscs_options);
    bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);

    std::vector<uint32_t> run_mailbox_init_val = {INIT_VALUE};

    std::function<void(uint64_t)> initialize_and_check_run_mailbox = [&](uint64_t run_mailbox_address_) {
        write_hex_vec_to_core(cluster, chip_id, core, run_mailbox_init_val, run_mailbox_address_);
        std::vector<uint32_t> run_mailbox_init_val_check;
        run_mailbox_init_val_check = read_hex_vec_from_core(
            cluster, chip_id, core, run_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t
        TT_ASSERT(run_mailbox_init_val_check[0] == INIT_VALUE);
        log_debug(
            tt::LogLLRuntime,
            "checked test_mailbox is correctly initialized to value = {} for core {}",
            run_mailbox_init_val_check[0],
            core.str());
    };

    initialize_and_check_run_mailbox(MEM_RUN_MAILBOX_ADDRESS);

    if (!involves_ncrisc) {
        disable_ncrisc(cluster, chip_id, core);
    } else {
        enable_ncrisc(cluster, chip_id, core);
    }

    if (!involves_triscs) {
        disable_triscs(cluster, chip_id, core);
    } else {
        enable_triscs(cluster, chip_id, core);
    }
}

void setup_riscs_on_specified_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const std::vector<CoreCoord> &cores) {
    for (const CoreCoord &core : cores) {
        setup_riscs_on_specified_core(cluster, chip_id, riscs_options, core);
    }
}

bool check_if_riscs_on_specified_core_done(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const CoreCoord &core) {

    std::function<bool(uint64_t)> get_mailbox_is_done = [&](uint64_t run_mailbox_address_) {
        std::vector<uint32_t> run_mailbox_read_val = {0};
        run_mailbox_read_val = read_hex_vec_from_core(
            cluster, chip_id, core, run_mailbox_address_, sizeof(uint32_t));  // read a single uint32_t

        if (run_mailbox_read_val[0] != INIT_VALUE && run_mailbox_read_val[0] != DONE_VALUE) {
            fprintf(stderr, "Read unexpected run_mailbox value: %x (expected %x or %x)\n", run_mailbox_read_val[0], INIT_VALUE, DONE_VALUE);
            TT_ASSERT(
                run_mailbox_read_val[0] == INIT_VALUE || run_mailbox_read_val[0] == DONE_VALUE);
        }

        return run_mailbox_read_val[0] == DONE_VALUE;
    };

    return get_mailbox_is_done(RUN_MAILBOX_ADDR);
}

void cleanup_risc_on_specified_core(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_options, const CoreCoord &core) {
    bool involves_triscs = deduce_if_involves_triscs(riscs_options);
    bool involves_ncrisc = deduce_if_involves_ncrisc(riscs_options);

    if (!involves_ncrisc) {
        enable_ncrisc(cluster, chip_id, core);
    }

    if (!involves_triscs) {
        enable_triscs(cluster, chip_id, core);
    }
}

void run_riscs_on_specified_cores(
    tt_cluster *cluster, int chip_id, const TensixRiscsOptions riscs_option, const std::vector<CoreCoord> &cores, const std::vector<uint32_t> &hugepage_done_addrs, bool stagger_start) {

    bool write_to_huge_page = hugepage_done_addrs.size() > 0;
    if (write_to_huge_page) {
        uint32_t dispatch_done_addr = 0;
        vector<uint32_t> reset = {0};
        cluster->write_sysmem_vec(reset, dispatch_done_addr, chip_id);
    }

    for (const CoreCoord &core_ : cores) {
        tt_cxy_pair core = tt_cxy_pair(chip_id, core_);
        if (stagger_start){
            cluster->set_remote_tensix_risc_reset(core, TENSIX_DEASSERT_SOFT_RESET);
        }
        else
        {
            cluster->set_remote_tensix_risc_reset(core, TENSIX_DEASSERT_SOFT_RESET_NO_STAGGER);
        }
    }

    if (write_to_huge_page) {
        // In this path, host polls hugepage memory rather than the cores
        // to check that they're done
        bool riscs_are_done = false;
        uint32_t dispatch_done_addr = 0;
        vector<uint32_t> reset = {0};

        vector<uint32_t> riscs_are_done_vec;
        while (not riscs_are_done) {
            riscs_are_done = true;
            // Poll hugepage to see that dispatch has completed
            uint32_t idx = 0;
            for (const CoreCoord &core : cores) {
                uint32_t hugepage_done_addr = hugepage_done_addrs.at(idx++);
                cluster->read_sysmem_vec(riscs_are_done_vec, dispatch_done_addr, 4, chip_id);
                riscs_are_done &= riscs_are_done_vec.at(0) == NOTIFY_HOST_KERNEL_COMPLETE_VALUE;
            }
        }
        cluster->write_sysmem_vec(reset, dispatch_done_addr, chip_id);
    } else {
        // In this path, host polls core L1 to check whether they're done
        bool riscs_are_done = false;
        while (!riscs_are_done) {
            riscs_are_done = true;
            for (const CoreCoord &core : cores) {
                riscs_are_done &= check_if_riscs_on_specified_core_done(cluster, chip_id, riscs_option, core);
            }
        }
    }

    for (const CoreCoord &core : cores) {
        cleanup_risc_on_specified_core(cluster, chip_id, riscs_option, core);
    }

    assert_reset_for_all_chips(cluster);
}


}  // namespace internal_

}  // namespace llrt

}  // namespace tt
