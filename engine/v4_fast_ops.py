"""Small CUDA helpers for V4 decode.

* ``gather_bytes`` copies G selected expert blobs with a device index (no host sync).
* ``oneshot_allreduce_`` sums a small contiguous fp32 vector across TP ranks with
  one kernel and CUDA IPC, instead of an NCCL ring launch per tensor.

Both launch on the current stream so a CUDA graph can capture them.
Does not import sglang or vllm.
"""

from __future__ import annotations

import os
from typing import Optional

# Must run before torch initializes the CUDA context. torchrun does not set this.
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

import torch
import torch.distributed as dist

_OPS = None
_OPS_ERR: Optional[str] = None
_AR = None
_ORIG_AR = None


def _load_ops():
    global _OPS, _OPS_ERR
    if _OPS is not None or _OPS_ERR is not None:
        return _OPS
    # Eight ranks compiling load_inline at once race the build directory.
    # Rank 0 compiles; the others wait and then load the cached extension.
    multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if multi and dist.get_rank() != 0:
        dist.barrier()
    try:
        return _load_ops_impl()
    finally:
        if multi and dist.get_rank() == 0:
            dist.barrier()


def _load_ops_impl():
    global _OPS, _OPS_ERR
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:  # pragma: no cover
        _OPS_ERR = f"cpp_extension: {exc}"
        return None

    cpp = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

extern "C" void gather_bytes_cuda(const int64_t* ptrs, const int64_t* idx, void* dst, int64_t nbytes, int groups, cudaStream_t stream);
extern "C" void oneshot_allreduce_cuda(float* dst, const int64_t* pubs, const int64_t* flags, const int64_t* dones, int* counter, int rank, int world, int n, cudaStream_t stream);

void gather_bytes(at::Tensor ptrs, at::Tensor idx, at::Tensor dst, int64_t nbytes) {
  TORCH_CHECK(ptrs.is_cuda() && idx.is_cuda() && dst.is_cuda(), "gather tensors must be CUDA");
  TORCH_CHECK(ptrs.dtype() == torch::kInt64 && idx.dtype() == torch::kInt64, "ptrs/idx must be int64");
  TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
  gather_bytes_cuda(
      ptrs.data_ptr<int64_t>(),
      idx.data_ptr<int64_t>(),
      dst.data_ptr(),
      nbytes,
      static_cast<int>(idx.size(0)),
      c10::cuda::getCurrentCUDAStream().stream());
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gather_bytes: ", cudaGetErrorString(err));
}

void oneshot_allreduce(at::Tensor dst, at::Tensor pubs, at::Tensor flags, at::Tensor dones, at::Tensor counter, int64_t rank) {
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dst");
  TORCH_CHECK(dst.dtype() == torch::kFloat32, "oneshot dst must be fp32");
  TORCH_CHECK(pubs.is_cuda() && flags.is_cuda() && dones.is_cuda() && counter.is_cuda(), "ipc tables");
  oneshot_allreduce_cuda(
      dst.data_ptr<float>(),
      pubs.data_ptr<int64_t>(),
      flags.data_ptr<int64_t>(),
      dones.data_ptr<int64_t>(),
      counter.data_ptr<int>(),
      static_cast<int>(rank),
      static_cast<int>(pubs.size(0)),
      static_cast<int>(dst.numel()),
      c10::cuda::getCurrentCUDAStream().stream());
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "oneshot_allreduce: ", cudaGetErrorString(err));
}

std::vector<int64_t> ipc_handle_of(int64_t ptr) {
  cudaIpcMemHandle_t handle;
  cudaError_t err = cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr));
  TORCH_CHECK(err == cudaSuccess, "cudaIpcGetMemHandle: ", cudaGetErrorString(err));
  std::vector<int64_t> out(8, 0);
  static_assert(sizeof(handle) <= 64, "ipc handle");
  std::memcpy(out.data(), &handle, sizeof(handle));
  return out;
}

int64_t ipc_open(std::vector<int64_t> raw) {
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, raw.data(), sizeof(handle));
  void* ptr = nullptr;
  cudaError_t err = cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess) {
    cudaGetLastError();
    TORCH_CHECK(false, "cudaIpcOpenMemHandle: ", cudaGetErrorString(err));
  }
  return reinterpret_cast<int64_t>(ptr);
}

void clear_cuda_error() {
  cudaGetLastError();
}

int64_t host_register(int64_t ptr, int64_t nbytes) {
  void* host = reinterpret_cast<void*>(ptr);
  cudaError_t err = cudaHostRegister(
      host, static_cast<size_t>(nbytes), cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err == cudaErrorHostMemoryAlreadyRegistered) {
    cudaGetLastError();
    err = cudaSuccess;
  }
  TORCH_CHECK(err == cudaSuccess, "cudaHostRegister: ", cudaGetErrorString(err));
  void* dev = nullptr;
  err = cudaHostGetDevicePointer(&dev, host, 0);
  TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer: ", cudaGetErrorString(err));
  return reinterpret_cast<int64_t>(dev);
}

at::Tensor raw_alloc(int64_t nbytes) {
  TORCH_CHECK(nbytes > 0, "nbytes");
  void* ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, static_cast<size_t>(nbytes));
  TORCH_CHECK(err == cudaSuccess, "cudaMalloc: ", cudaGetErrorString(err));
  err = cudaMemset(ptr, 0, static_cast<size_t>(nbytes));
  TORCH_CHECK(err == cudaSuccess, "cudaMemset: ", cudaGetErrorString(err));
  auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
  return torch::from_blob(
      ptr,
      {nbytes},
      [](void* p) { cudaFree(p); },
      opts);
}
"""
    cuda = r"""
#include <cuda_runtime.h>
#include <cstdint>

__global__ void gather_bytes_k(const uint8_t* const* ptrs, const int64_t* idx, uint8_t* dst, int nbytes, int groups) {
  int g = blockIdx.y;
  if (g >= groups) return;
  int64_t src_i = idx[g];
  const uint8_t* src = ptrs[src_i];
  uint8_t* out = dst + static_cast<int64_t>(g) * nbytes;
  bool vec = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(out) | static_cast<unsigned>(nbytes)) & 15u) == 0u;
  if (vec) {
    int n4 = nbytes >> 4;
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(out);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
      d4[i] = s4[i];
    }
  } else {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nbytes; i += blockDim.x * gridDim.x) {
      out[i] = src[i];
    }
  }
}

extern "C" void gather_bytes_cuda(const int64_t* ptrs, const int64_t* idx, void* dst, int64_t nbytes, int groups, cudaStream_t stream) {
  int threads = 256;
  int blocks = static_cast<int>(((nbytes >> 4) + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 1024) blocks = 1024;
  gather_bytes_k<<<dim3(blocks, groups), threads, 0, stream>>>(
      reinterpret_cast<const uint8_t* const*>(ptrs),
      idx,
      reinterpret_cast<uint8_t*>(dst),
      static_cast<int>(nbytes),
      groups);
}

__global__ void oneshot_f32_k(
    float* dst,
    const float* const* pubs,
    int* const* flags,
    int* const* dones,
    int* counter,
    int rank,
    int world,
    int n) {
  // Each thread fences its own stores. A fence on thread 0 alone does not
  // publish the other threads. dones[r] == seq means rank r has finished
  // reading every payload of that step.
  __shared__ int seq_s;
  if (threadIdx.x == 0) {
    int seq = atomicAdd(counter, 1) + 1;
    seq_s = seq;
    if (seq > 1) {
      for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        while (atomicAdd_system(dones[r], 0) < seq - 1) {
        }
      }
    }
  }
  __syncthreads();
  float* mine = const_cast<float*>(pubs[rank]);
  bool vec = ((n & 3) == 0) && ((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(mine)) & 15u) == 0u;
  if (vec) {
    int n4 = n >> 2;
    float4* m4 = reinterpret_cast<float4*>(mine);
    const float4* d4 = reinterpret_cast<const float4*>(dst);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      m4[i] = d4[i];
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      mine[i] = dst[i];
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    int seq = seq_s;
    atomicExch_system(flags[rank], seq);
    for (int r = 0; r < world; ++r) {
      while (atomicAdd_system(flags[r], 0) < seq) {
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  if (vec) {
    int n4 = n >> 2;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int r = 0; r < world; ++r) {
        float4 v = reinterpret_cast<const float4*>(pubs[r])[i];
        acc.x += v.x;
        acc.y += v.y;
        acc.z += v.z;
        acc.w += v.w;
      }
      reinterpret_cast<float4*>(dst)[i] = acc;
    }
  } else {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      float acc = 0.f;
      for (int r = 0; r < world; ++r) {
        acc += pubs[r][i];
      }
      dst[i] = acc;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomicExch_system(dones[rank], seq_s);
  }
}

extern "C" void oneshot_allreduce_cuda(float* dst, const int64_t* pubs, const int64_t* flags, const int64_t* dones, int* counter, int rank, int world, int n, cudaStream_t stream) {
  oneshot_f32_k<<<1, 256, 0, stream>>>(
      dst,
      reinterpret_cast<const float* const*>(pubs),
      reinterpret_cast<int* const*>(flags),
      reinterpret_cast<int* const*>(dones),
      counter,
      rank,
      world,
      n);
}
"""
    try:
        # no_implicit_headers: load_inline otherwise prepends torch/types.h,
        # and nvcc 13.3 fails inside ATen's List_inl.h. The .cu includes only
        # cuda_runtime.h. The C++ file includes torch/extension.h itself.
        _OPS = load_inline(
            name="sglang_lite_v4_fast_ops_v5",
            cpp_sources=[cpp],
            cuda_sources=[cuda],
            functions=[
                "gather_bytes",
                "oneshot_allreduce",
                "ipc_handle_of",
                "ipc_open",
                "raw_alloc",
                "clear_cuda_error",
                "host_register",
            ],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            with_cuda=True,
            no_implicit_headers=True,
            verbose=False,
        )
    except Exception as exc:
        _OPS_ERR = str(exc)
        return None
    return _OPS


def ops_error() -> Optional[str]:
    _load_ops()
    return _OPS_ERR


def gather_bytes(ptrs: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy ``ptrs[idx[g]]`` into ``dst[g]`` for each group. ``dst[0].nbytes`` is the blob size."""
    mod = _load_ops()
    if mod is None:
        raise RuntimeError(f"fast ops unavailable: {_OPS_ERR}")
    nbytes = int(dst[0].numel()) * int(dst.element_size())
    mod.gather_bytes(ptrs, idx.contiguous(), dst, nbytes)


class OneShotAllReduce:
    """In-place sum of a small fp32 CUDA tensor across the TP process group."""

    def __init__(self, max_n: int = 16384):
        mod = _load_ops()
        if mod is None:
            raise RuntimeError(f"fast ops unavailable: {_OPS_ERR}")
        self.mod = mod
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.max_n = max_n
        self.backend = "ipc"
        ipc_ok = False
        try:
            self._init_ipc()
            ipc_ok = True
        except Exception:
            mod.clear_cuda_error()
            ipc_ok = False
        bad = torch.tensor([0 if ipc_ok else 1], device="cuda", dtype=torch.int32)
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        if int(bad.item()) != 0:
            # Host-mapped shared memory was tried on this 8×5090 box. The
            # mapping registers, then the system-atomic spin never observes
            # another GPU's flag, so the kernel hangs. NCCL stays.
            raise RuntimeError("peer access unavailable")

    def _set_ptrs(self, pubs: list[int], flags: list[int], dones: list[int]) -> None:
        self.pubs = torch.tensor(pubs, dtype=torch.int64, device="cuda")
        self.flags = torch.tensor(flags, dtype=torch.int64, device="cuda")
        self.dones = torch.tensor(dones, dtype=torch.int64, device="cuda")

    def _init_ipc(self) -> None:
        mod = self.mod
        pub = mod.raw_alloc(self.max_n * 4)
        flag = mod.raw_alloc(4)
        done = mod.raw_alloc(4)
        counter = mod.raw_alloc(4)
        self._keep = (pub, flag, done, counter)
        self.counter = counter.view(torch.int32)
        # NCCL cannot gather CPU tensors. Handles are 8 int64s; this is init only.
        h_pub = torch.tensor(mod.ipc_handle_of(pub.data_ptr()), dtype=torch.int64, device="cuda")
        h_flag = torch.tensor(mod.ipc_handle_of(flag.data_ptr()), dtype=torch.int64, device="cuda")
        h_done = torch.tensor(mod.ipc_handle_of(done.data_ptr()), dtype=torch.int64, device="cuda")
        pub_hs = [torch.empty_like(h_pub) for _ in range(self.world)]
        flag_hs = [torch.empty_like(h_flag) for _ in range(self.world)]
        done_hs = [torch.empty_like(h_done) for _ in range(self.world)]
        dist.all_gather(pub_hs, h_pub)
        dist.all_gather(flag_hs, h_flag)
        dist.all_gather(done_hs, h_done)
        pubs: list[int] = []
        flags: list[int] = []
        dones: list[int] = []
        for r in range(self.world):
            if r == self.rank:
                pubs.append(pub.data_ptr())
                flags.append(flag.data_ptr())
                dones.append(done.data_ptr())
            else:
                pubs.append(int(mod.ipc_open(pub_hs[r].tolist())))
                flags.append(int(mod.ipc_open(flag_hs[r].tolist())))
                dones.append(int(mod.ipc_open(done_hs[r].tolist())))
        self._set_ptrs(pubs, flags, dones)

    def _init_shm(self) -> None:
        """Do not call. Host-mapped system atomics hang on this 8×5090 box."""
        raise RuntimeError("shm oneshot hangs on 8x 5090 without peer access")
        import ctypes
        import mmap

        mod = self.mod
        nbytes = self.world * self.max_n * 4 + self.world * 8
        nbytes = (nbytes + 4095) // 4096 * 4096
        path = "/dev/shm/sglang_lite_v4_oneshot"
        if self.rank == 0:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
            os.ftruncate(fd, nbytes)
            mm = mmap.mmap(fd, nbytes)
            mm.seek(0)
            mm.write(b"\x00" * nbytes)
        dist.barrier()
        if self.rank != 0:
            fd = os.open(path, os.O_RDWR)
            mm = mmap.mmap(fd, nbytes)
        dist.barrier()
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        dev = int(mod.host_register(addr, nbytes))
        counter = mod.raw_alloc(4)
        self._keep = (mm, fd, counter)
        self.counter = counter.view(torch.int32)
        pubs: list[int] = []
        flags: list[int] = []
        dones: list[int] = []
        flag_base = self.world * self.max_n * 4
        done_base = flag_base + self.world * 4
        for r in range(self.world):
            pubs.append(dev + r * self.max_n * 4)
            flags.append(dev + flag_base + r * 4)
            dones.append(dev + done_base + r * 4)
        self._set_ptrs(pubs, flags, dones)
        self.backend = "shm"

    def __call__(self, flat_fp32: torch.Tensor) -> None:
        n = int(flat_fp32.numel())
        if n > self.max_n:
            raise RuntimeError(f"oneshot numel {n} > {self.max_n}")
        self.mod.oneshot_allreduce(flat_fp32, self.pubs, self.flags, self.dones, self.counter, self.rank)


def uninstall_oneshot_allreduce() -> None:
    """Put NCCL back. Safe to call when one-shot was never installed."""
    global _AR, _ORIG_AR
    if _ORIG_AR is not None:
        dist.all_reduce = _ORIG_AR  # type: ignore[assignment]
    _ORIG_AR = None
    _AR = None


def install_oneshot_allreduce(max_n: int = 16384) -> bool:
    """Replace ``torch.distributed.all_reduce`` for small contiguous fp32 CUDA sums.

    bf16 embedding reductions stay on NCCL so the sum rounding matches the
    eager baseline. Returns False if the IPC buffer cannot be opened.
    """
    global _AR, _ORIG_AR
    if _AR is not None:
        return True
    if not dist.is_initialized() or dist.get_world_size() < 2:
        return False
    try:
        _AR = OneShotAllReduce(max_n=max_n)
    except Exception as exc:
        print(f"[sglang-lite] one-shot allreduce off: {exc}", flush=True)
        _AR = None
        return False
    _ORIG_AR = dist.all_reduce

    def fast_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        use = (
            _AR is not None
            and op == dist.ReduceOp.SUM
            and group is None
            and not async_op
            and torch.is_tensor(tensor)
            and tensor.is_cuda
            and tensor.dtype == torch.float32
            and tensor.numel() <= _AR.max_n
        )
        if not use:
            return _ORIG_AR(tensor, op=op, group=group, async_op=async_op)
        if tensor.is_contiguous():
            _AR(tensor.reshape(-1))
            return None
        tmp = tensor.contiguous().reshape(-1)
        _AR(tmp)
        tensor.copy_(tmp.view(tensor.shape))
        return None

    dist.all_reduce = fast_all_reduce  # type: ignore[assignment]
    if dist.get_rank() == 0:
        print(f"[sglang-lite] one-shot allreduce on ({_AR.backend}, max_n={max_n})", flush=True)
    return True


def oneshot_self_test() -> bool:
    """Compare one small sum against NCCL, then require a speed win. All ranks must call."""
    if not dist.is_initialized() or dist.get_world_size() < 2:
        return False
    n = 4096
    src = torch.randn(n, device="cuda", dtype=torch.float32) + dist.get_rank()
    ref = src.clone()
    dist.all_reduce(ref)
    ok_install = install_oneshot_allreduce()
    if not ok_install:
        return False
    got = src.clone()
    dist.all_reduce(got)
    err = (got - ref).abs().max().item()
    # Warm both paths. NCCL is the original; oneshot is installed, so time via _AR directly.
    ar = _AR
    assert ar is not None
    trial = src.clone()
    for _ in range(5):
        trial.copy_(src)
        ar(trial)
    torch.cuda.synchronize()
    dist.barrier()
    iters = 40
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    buf = src.clone()
    start.record()
    for _ in range(iters):
        for _k in range(87):
            ar(buf)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    # int32 stays on NCCL (the patch is fp32-only), so this vote cannot deadlock
    # inside the one-shot kernel. Any rank that is wrong or slow rejects it.
    bad = torch.tensor(
        [1 if (err > 1e-2 or ms > 6.0) else 0], device="cuda", dtype=torch.int32
    )
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    reject = int(bad.item()) != 0
    if dist.get_rank() == 0:
        print(
            f"[sglang-lite] oneshot {ar.backend} max_abs {err:.3e} step87_ms {ms:.3f}",
            flush=True,
        )
    dist.barrier()
    if reject:
        uninstall_oneshot_allreduce()
        if dist.get_rank() == 0:
            print("[sglang-lite] one-shot allreduce rejected, NCCL stays", flush=True)
        return False
    return True


def main() -> int:
    """8-GPU self-test. Exit 0 when one-shot stays installed, 1 when NCCL stays."""
    import torch.distributed as dist_mod

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist_mod.init_process_group("nccl")
    torch.cuda.set_device(0)
    ok = False
    try:
        ok = oneshot_self_test()
    except Exception as exc:
        uninstall_oneshot_allreduce()
        if rank == 0:
            print(f"[sglang-lite] one-shot allreduce off: {exc}", flush=True)
        ok = False
    if world > 1:
        dist_mod.barrier()
        dist_mod.destroy_process_group()
    if rank == 0:
        print("ONESHOT_OK" if ok else "ONESHOT_REJECT", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
