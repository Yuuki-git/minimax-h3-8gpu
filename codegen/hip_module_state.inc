struct ModuleSlot {
    std::mutex mutex;
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    unsigned max_lds = 0;
};

static ModuleSlot modules[8];
static thread_local const char *launch_phase = "not_called";

static hipError_t configure_lds_limit(int device, unsigned limit) {
    if (device < 0 || device >= 8) {
        return hipErrorInvalidDevice;
    }
    auto &slot = modules[device];
    std::lock_guard<std::mutex> lock(slot.mutex);
    if (limit == 0 || (slot.function && slot.max_lds != limit)) {
        return hipErrorInvalidValue;
    }
    slot.max_lds = limit;
    return hipSuccess;
}

static hipError_t function_for_device(hipFunction_t *function) {
    int device;
    launch_phase = "hipGetDevice";
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
        return status;
    }
    if (device < 0 || device >= 8) {
        launch_phase = "device_index";
        return hipErrorInvalidDevice;
    }
    auto &slot = modules[device];
    std::lock_guard<std::mutex> lock(slot.mutex);
    if (!slot.function) {
        if constexpr (required_lds > 0) {
            if (slot.max_lds == 0) {
                launch_phase = "lds_limit_uninitialized";
                return hipErrorInvalidValue;
            }
            if (slot.max_lds < required_lds) {
                launch_phase = "lds_limit";
                return hipErrorInvalidValue;
            }
        }
        hipModule_t module;
        launch_phase = "hipModuleLoadData";
        status = hipModuleLoadData(&module, image);
        if (status != hipSuccess) {
            return status;
        }
        hipFunction_t kernel;
        launch_phase = "hipModuleGetFunction";
        status = hipModuleGetFunction(&kernel, module, entry);
        if (status != hipSuccess) {
            (void)hipModuleUnload(module);
            return status;
        }
        slot.module = module;
        slot.function = kernel;
    }
    *function = slot.function;
    return hipSuccess;
}
