"""Match CUDA and EGL devices by their driver-reported identities."""
import ctypes
import os
from pathlib import Path


def enumerate_devices(expected_cuda_devices=3):
    """Obtain GPU IDs with the worker's exact CUDA mask, in a fresh process."""
    import ctypes as C
    ptr, integer, boolean = C.c_void_p, C.c_int, C.c_uint
    pi = C.POINTER(integer)
    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    egl.eglGetProcAddress.restype = ptr
    def proc(name, output, *args):
        address = egl.eglGetProcAddress(name.encode())
        return C.CFUNCTYPE(output, *args)(address) if address else None
    devices = proc("eglQueryDevicesEXT", boolean, integer, C.POINTER(ptr), pi)
    attrib = proc("eglQueryDeviceAttribEXT", boolean, ptr, integer, C.POINTER(C.c_ssize_t))
    string = proc("eglQueryDeviceStringEXT", C.c_char_p, ptr, integer)
    binary = proc("eglQueryDeviceBinaryEXT", boolean, ptr, integer, integer, ptr, pi)
    count = integer()
    if not devices or not attrib or not string or not devices(0, None, C.byref(count)) or count.value < expected_cuda_devices:
        raise RuntimeError("EGL device enumeration/attribute queries unavailable")
    handles = (ptr * count.value)()
    if not devices(len(handles), handles, C.byref(count)):
        raise RuntimeError("EGL device enumeration failed")
    egl_rows = []
    for index, device in enumerate(handles[:count.value]):
        extensions = (string(device, 0x3055) or b"").split()
        ordinal = C.c_ssize_t(-1)
        gpu_uuid = (C.c_ubyte * 16)()
        size = integer()
        row = {"egl_index": index, "cuda_ordinal": None, "uuid": None, "pci": None}
        if b"EGL_NV_device_cuda" in extensions and attrib(device, 0x323A, C.byref(ordinal)):
            row["cuda_ordinal"] = ordinal.value
        if binary and b"EGL_EXT_device_persistent_id" in extensions:
            if binary(device, 0x335C, 16, gpu_uuid, C.byref(size)) and size.value == 16:
                row["uuid"] = bytes(gpu_uuid).hex()
        for ext, key in [(b"EGL_EXT_device_drm_render_node", 0x3377),(b"EGL_EXT_device_drm", 0x3233)]:
            node = string(device,key) if ext in extensions else None
            if node:
                device_link = Path("/sys/class/drm") / Path(os.fsdecode(node)).name / "device"
                if device_link.exists():
                    row["pci"] = device_link.resolve().name.lower()
                    break
        egl_rows.append(row)
    cuda = C.CDLL("libcuda.so.1")
    def check(ret):
        if ret != 0:
            raise RuntimeError(f"CUDA device identification failed ({ret})")
    check(cuda.cuInit(0))
    check(cuda.cuDeviceGetCount(C.byref(count)))
    if count.value != expected_cuda_devices:
        raise RuntimeError(f"Expected {expected_cuda_devices} visible CUDA devices, found {count.value}")
    cuda_rows = []
    for rank in range(count.value):
        handle, bus, gpu_uuid = integer(), C.create_string_buffer(40), (C.c_ubyte * 16)()
        check(cuda.cuDeviceGet(C.byref(handle), rank))
        check(cuda.cuDeviceGetPCIBusId(bus, len(bus), handle))
        check(cuda.cuDeviceGetUuid(C.byref(gpu_uuid), handle))
        cuda_rows.append({"rank":rank, "uuid":bytes(gpu_uuid).hex(), "pci":bus.value.decode().lower()})
    return {"egl": egl_rows, "cuda": cuda_rows}


def rank_map(info, *, require_identity=False):
    selected = {}
    for gpu in info["cuda"]:
        matches = []
        for row in info["egl"]:
            if row["cuda_ordinal"] is None or row["cuda_ordinal"] < 0:
                continue
            identity = [key for key in ("uuid","pci") if row[key] is not None]
            if identity and all(row[key] == gpu[key] for key in identity):
                matches.append({**row, "identity_verified_by":identity})
        if not matches and not require_identity:
            # The extension returns the CUDA ordinal in this process. Only use
            # this fallback when the complete EGL ordinal set fits this mask.
            reported = [r["cuda_ordinal"] for r in info["egl"] if r["cuda_ordinal"] is not None and r["cuda_ordinal"] >= 0]
            if sorted(reported) == list(range(len(info["cuda"]))):
                matches = [{**r, "identity_verified_by":["same-process CUDA ordinal only"]}
                           for r in info["egl"] if r["cuda_ordinal"] == gpu["rank"] and not r["uuid"] and not r["pci"]]
        if len(matches) != 1:
            raise RuntimeError(f"Ambiguous/unavailable EGL identity for rank {gpu['rank']}: {len(matches)} matches")
        selected[gpu["rank"]] = matches[0]
    if len({r["egl_index"] for r in selected.values()}) != len(info["cuda"]):
        raise RuntimeError("EGL devices do not have a unique mapping to CUDA workers")
    return selected


def active_egl_device():
    # EGL_EXT_device_query / EGL_NV_device_cuda are optional driver extensions.
    egl = ctypes.CDLL("libEGL.so.1")
    ptr = ctypes.c_void_p
    egl.eglGetCurrentDisplay.restype = ptr
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ptr
    display = egl.eglGetCurrentDisplay()
    addr = egl.eglGetProcAddress(b"eglQueryDisplayAttribEXT")
    devaddr = egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    if not (display and addr and devaddr):
        return None
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ptr, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    device, ordinal = ctypes.c_ssize_t(), ctypes.c_ssize_t()
    if query(addr)(display, 0x322C, ctypes.byref(device)) and query(devaddr)(device.value, 0x323A, ctypes.byref(ordinal)):
        return ordinal.value
    return None
