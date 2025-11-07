import logging
from os import environ
from sys import stdout
from typing import Generator, List, Literal, NamedTuple, Optional, Sequence, Tuple, Type, Union, get_args

import numpy
from numpy.typing import NDArray, DTypeLike
from mpi4py import MPI
from mpi4py.util import dtlib

GridMapType = Literal["default", "reversed", "shared"]
from .array import BackendType, cudaDeviceAPI


class _MPILogger:
    def __init__(self, root: int = 0) -> None:
        self.root = root
        formatter = logging.Formatter(fmt="{name} {levelname}: {message}", style="{")
        stdout_handler = logging.StreamHandler(stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        self.logger = logging.getLogger("PyQUDA")
        self.logger.level = logging.DEBUG
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Type[Warning]):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg), stack_info=True)

    def error(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Type[Exception]):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


class _ComputeCapability(NamedTuple):
    major: int
    minor: int


_MPI_LOGGER: _MPILogger = _MPILogger()
_MPI_COMM: MPI.Intracomm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_MAP: GridMapType = "default"
"""For MPI, the default node mapping is lexicographical with t varying fastest."""
_GRID_SIZE: Optional[Tuple[int, ...]] = None
_GRID_COORD: Optional[Tuple[int, ...]] = None
_SHARED_RANK_LIST: Optional[List[int]] = None
_CUDA_BACKEND: BackendType = "cupy"
_CUDA_IS_HIP: bool = False
_CUDA_DEVICE: int = -1
_CUDA_COMPUTE_CAPABILITY: _ComputeCapability = _ComputeCapability(0, 0)


def _defaultRankFromCoord(coords: Sequence[int], dims: Sequence[int]) -> int:
    rank = 0
    for coord, dim in zip(coords, dims):
        rank = rank * dim + coord
    return rank


def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]


def getRankFromCoord(grid_coord: List[int]) -> int:
    grid_size = getGridSize()
    if len(grid_coord) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Grid coordinate {grid_coord} and grid size {grid_size} must have the same dimension",
            ValueError,
        )

    if _GRID_MAP == "default":
        mpi_rank = _defaultRankFromCoord(grid_coord, grid_size)
    elif _GRID_MAP == "reversed":
        mpi_rank = _defaultRankFromCoord(grid_coord[::-1], grid_size[::-1])
    elif _GRID_MAP == "shared":
        assert _SHARED_RANK_LIST is not None
        mpi_rank = _SHARED_RANK_LIST.index(_defaultRankFromCoord(grid_coord, grid_size))
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)
    return mpi_rank


def getCoordFromRank(mpi_rank: int) -> List[int]:
    grid_size = getGridSize()

    if _GRID_MAP == "default":
        grid_coord = _defaultCoordFromRank(mpi_rank, grid_size)
    elif _GRID_MAP == "reversed":
        grid_coord = _defaultCoordFromRank(mpi_rank, grid_size[::-1])[::-1]
    elif _GRID_MAP == "shared":
        assert _SHARED_RANK_LIST is not None
        grid_coord = _defaultCoordFromRank(_SHARED_RANK_LIST[mpi_rank], grid_size)
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)
    return grid_coord


def getNeighbourRank():
    grid_size = getGridSize()
    grid_coord = getGridCoord()

    neighbour_forward = []
    neighbour_backward = []
    for d in range(len(grid_size)):
        g, G = grid_coord[d], grid_size[d]
        grid_coord[d] = (g + 1) % G
        neighbour_forward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = (g - 1) % G
        neighbour_backward.append(getRankFromCoord(grid_coord))
        grid_coord[d] = g
    return neighbour_forward + neighbour_backward


def getSublatticeSize(latt_size: Sequence[int], force_even: bool = True):
    grid_size = getGridSize()
    if len(latt_size) != len(grid_size):
        _MPI_LOGGER.critical(
            f"Lattice size {latt_size} and grid size {grid_size} must have the same dimension",
            ValueError,
        )
    if force_even:
        if not all([(GL % (2 * G) == 0 or GL * G == 1) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}, "
                "and sublattice size must be even in all directions for consistant even-odd preconditioning, "
                "otherwise the lattice size and grid size for this direction must be 1",
                ValueError,
            )
    else:
        if not all([(GL % G == 0) for GL, G in zip(latt_size, grid_size)]):
            _MPI_LOGGER.critical(
                f"lattice size {latt_size} must be divisible by gird size {grid_size}",
                ValueError,
            )
    return [GL // G for GL, G in zip(latt_size, grid_size)]


def _composition(n: int, d: int):
    """
    Writing n as the sum of d natural numbers
    """
    addend: List[List[int]] = []
    i = [0 for _ in range(d - 1)] + [n] + [0]
    while i[0] <= n:
        addend.append([i[s] - i[s - 1] for s in range(d)])
        i[d - 2] += 1
        for s in range(d - 2, 0, -1):
            if i[s] == n + 1:
                i[s] = 0
                i[s - 1] += 1
        for s in range(1, d - 1, 1):
            if i[s] < i[s - 1]:
                i[s] = i[s - 1]
    return addend


def _factorization(k: int, d: int):
    """
    Writing k as the product of d positive numbers
    """
    prime_factor: List[List[List[int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([[p**a for a in addend] for addend in _composition(n, d)])
    if k != 1:
        prime_factor.append([[k**a for a in addend] for addend in _composition(1, d)])
    return prime_factor


def _partition(
    factor: Union[int, List[List[List[int]]]],
    sublatt_size: List[int],
    grid_size: Optional[List[int]] = None,
    idx: int = 0,
) -> Generator[List[int], None, None]:
    if idx == 0:
        assert isinstance(factor, int) and grid_size is None
        grid_size = [1 for _ in range(len(sublatt_size))]
        factor = _factorization(factor, len(sublatt_size))
    assert isinstance(factor, list) and grid_size is not None
    if idx == len(factor):
        yield grid_size
    else:
        for factor_size in factor[idx]:
            for L, x in zip(sublatt_size, factor_size):
                if L % x != 0:
                    break
            else:
                yield from _partition(
                    factor,
                    [L // f for L, f in zip(sublatt_size, factor_size)],
                    [G * f for G, f in zip(grid_size, factor_size)],
                    idx + 1,
                )


def getDefaultGrid(mpi_size: int, latt_size: Sequence[int], evenodd: bool = True):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0, "lattice volume must be divisible by MPI size"
    if evenodd:
        assert (
            Lx % 2 == 0 and Ly % 2 == 0 and Lz % 2 == 0 and Lt % 2 == 0
        ), "lattice size must be even in all directions for even-odd preconditioning"
        partition = _partition(mpi_size, [Lx // 2, Ly // 2, Lz // 2, Lt // 2])
    else:
        partition = _partition(mpi_size, [Lx, Ly, Lz, Lt])
    for grid_size in partition:
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    if min_grid == []:
        _MPI_LOGGER.critical(
            f"Cannot get the proper grid for lattice size {latt_size} with {mpi_size} MPI processes", ValueError
        )
    return min(min_grid)


def setSharedRankList(grid_size: Sequence[int]):
    global _SHARED_RANK_LIST
    shared_comm = _MPI_COMM.Split_type(MPI.COMM_TYPE_SHARED)
    shared_size = shared_comm.Get_size()
    shared_rank = shared_comm.Get_rank()
    shared_root = shared_comm.bcast(_MPI_RANK)
    node_rank = _MPI_COMM.allgather(shared_root).index(shared_root)
    assert _MPI_SIZE % shared_size == 0
    node_grid_size = [G for G in grid_size]
    shared_grid_size = [1 for _ in grid_size]
    dim, last_dim = 0, len(grid_size) - 1
    while shared_size > 1:
        for prime in [2, 3, 5]:
            if node_grid_size[dim] % prime == 0 and shared_size % prime == 0:
                node_grid_size[dim] //= prime
                shared_grid_size[dim] *= prime
                shared_size //= prime
                last_dim = dim
                break
        else:
            if last_dim == dim:
                _MPI_LOGGER.critical("GlobalSharedMemory::GetShmDims failed", ValueError)
        dim = (dim + 1) % len(grid_size)
    grid_coord = [
        n * S + s
        for n, S, s in zip(
            _defaultCoordFromRank(node_rank, node_grid_size),
            shared_grid_size,
            _defaultCoordFromRank(shared_rank, shared_grid_size),
        )
    ]
    _SHARED_RANK_LIST = _MPI_COMM.allgather(_defaultRankFromCoord(grid_coord, grid_size))


def initGrid(
    grid_map: GridMapType = "default",
    grid_size: Optional[Sequence[int]] = None,
    latt_size: Optional[Sequence[int]] = None,
    evenodd: bool = True,
):
    global _GRID_MAP, _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        if grid_map not in get_args(GridMapType):
            _MPI_LOGGER.critical(f"Unsupported grid mapping {grid_map}", ValueError)
        _GRID_MAP = grid_map

        if grid_size is None and latt_size is not None:
            grid_size = getDefaultGrid(_MPI_SIZE, latt_size, evenodd)
        if grid_size is None:
            grid_size = [1, 1, 1, 1]

        if grid_map == "shared":
            setSharedRankList(grid_size)

        _GRID_SIZE = tuple(grid_size)
        _GRID_COORD = tuple(getCoordFromRank(_MPI_RANK))
        _MPI_LOGGER.info(f"Using the grid size {_GRID_SIZE}")
    else:
        _MPI_LOGGER.warning("Grid is already initialized", RuntimeWarning)


def initDevice(backend: BackendType = "cupy", device: int = -1, enable_mps: bool = False):
    global _CUDA_BACKEND, _CUDA_IS_HIP, _CUDA_DEVICE, _CUDA_COMPUTE_CAPABILITY
    if _CUDA_DEVICE < 0:
        from platform import node as gethostname

        if backend not in get_args(BackendType):
            _MPI_LOGGER.critical(f"Unsupported CUDA backend {backend}", ValueError)
        _CUDA_BACKEND = backend
        cudaGetDeviceCount, cudaGetDeviceProperties, cudaSetDevice, _CUDA_IS_HIP = cudaDeviceAPI(backend)
        _MPI_LOGGER.info(f"Using CUDA backend {backend}")

        # quda/include/communicator_quda.h
        # determine which GPU this rank will use
        hostname = gethostname()
        hostname_recv_buf = _MPI_COMM.allgather(hostname)

        if device < 0:
            device_count = cudaGetDeviceCount()
            if device_count == 0:
                _MPI_LOGGER.critical("No devices found", RuntimeError)

            # We initialize gpuid if it's still negative.
            device = 0
            for i in range(_MPI_RANK):
                if hostname == hostname_recv_buf[i]:
                    device += 1

            if device >= device_count:
                if enable_mps or environ.get("QUDA_ENABLE_MPS") == "1":
                    device %= device_count
                    print(f"MPS enabled, rank={_MPI_RANK:3d} -> gpu={device}")
                else:
                    _MPI_LOGGER.critical(f"Too few GPUs available on {hostname}", RuntimeError)
        _CUDA_DEVICE = device

        props = cudaGetDeviceProperties(device)
        if hasattr(props, "major") and hasattr(props, "minor"):
            _CUDA_COMPUTE_CAPABILITY = _ComputeCapability(int(props.major), int(props.minor))
        else:
            _CUDA_COMPUTE_CAPABILITY = _ComputeCapability(int(props["major"]), int(props["minor"]))

        cudaSetDevice(device)
    else:
        _MPI_LOGGER.warning("Device is already initialized", RuntimeWarning)


def isGridInitialized():
    return _GRID_SIZE is not None


def isDeviceInitialized():
    return _CUDA_DEVICE >= 0


def getLogger():
    return _MPI_LOGGER


def setLoggerLevel(level: Literal["debug", "info", "warning", "error", "critical"]):
    _MPI_LOGGER.logger.setLevel(level.upper())


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridMap():
    return _GRID_MAP


def getGridSize():
    if _GRID_SIZE is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return list(_GRID_SIZE)


def getGridCoord():
    if _GRID_COORD is None:
        _MPI_LOGGER.critical("Grid is not initialized", RuntimeError)
    return list(_GRID_COORD)


def getCUDABackend():
    return _CUDA_BACKEND


def isHIP():
    return _CUDA_IS_HIP


def getCUDADevice():
    return _CUDA_DEVICE


def getCUDAComputeCapability():
    return _CUDA_COMPUTE_CAPABILITY


def getSubarray(dtype: DTypeLike, shape: Sequence[int], axes: Sequence[int]):
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    grid = getGridSize()
    coord = getGridCoord()
    for j, i in enumerate(axes):
        sizes[i] *= grid[j]
        starts[i] *= coord[j]

    dtype_str = numpy.dtype(dtype).str
    native_dtype_str = dtype_str if not dtype_str.startswith(">") else dtype_str.replace(">", "<")
    return native_dtype_str, dtlib.from_numpy_dtype(native_dtype_str).Create_subarray(sizes, subsizes, starts)


# def readMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
#     native_dtype_str, filetype = getSubarray(dtype, shape, axes)
#     buf = numpy.empty(shape, native_dtype_str)

#     fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_RDONLY)
#     filetype.Commit()
#     fh.Set_view(disp=offset, filetype=filetype)
#     fh.Read_all(buf)
#     filetype.Free()
#     fh.Close()

#     return buf.view(dtype)

def readMPIFile(
    filename: str,
    dtype: DTypeLike,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
    io_procs: int,
    max_chunk_size: int = 2**30
) -> NDArray:
    """
    使用MPI聚合I/O策略并行读取文件。
    I/O进程负责读取其对应进程组的数据,然后通过Alltoallw分发。
    支持分块传输以处理 sdispls 和 sendcounts 超过 2GB 限制的情况。

    Parameters
    ----------
    filename:
        文件名。
    dtype:
        数据类型。
    offset:
        文件读取的起始偏移量。
    shape:
        每个进程期望读取的数据的本地形状 (local shape)。
    axes:
        需要跨进程分布的全局数组的轴。
    io_procs:
        用于I/O的聚合器进程数量。
    max_chunk_size:
        单次传输的最大字节数（默认1GB）。
    """
    comm = getMPIComm()
    rank = getMPIRank()
    grid = getGridSize()
    comm_size = getMPISize()

    if rank == 0:
        getLogger().info(f"Reading file {filename} with MPI aggregation I/O")

    io_procs = min(io_procs, comm_size)
    
    numpy_dtype = numpy.dtype(dtype)
    native_dtype_str = numpy_dtype.str.replace('>', '<') if numpy_dtype.str.startswith('>') else numpy_dtype.str
    native_dtype = numpy.dtype(native_dtype_str)
    itemsize = native_dtype.itemsize

    base_group_size = comm_size // io_procs
    last_group_start = (io_procs - 1) * base_group_size
    
    if rank < last_group_start:
        my_io_group = rank // base_group_size
        local_rank_in_group = rank % base_group_size
        procs_in_my_group = base_group_size
        my_io_leader = my_io_group * base_group_size
    else:
        my_io_group = io_procs - 1
        my_io_leader = last_group_start
        local_rank_in_group = rank - my_io_leader
        procs_in_my_group = comm_size - last_group_start
    
    is_io_proc = (local_rank_in_group == 0)
    
    local_data_size_bytes = int(numpy.prod(shape, dtype=numpy.int64) * itemsize)
    max_displacement = (procs_in_my_group - 1) * local_data_size_bytes

    # 计算安全的 chunk 大小，确保 sdispls 和 sendcounts/recvcounts 不超过 2GB 限制
    MPI_DISP_LIMIT = 2**31 - 1
    
    need_send_chunking = max_displacement >= MPI_DISP_LIMIT
    need_recv_chunking = local_data_size_bytes >= MPI_DISP_LIMIT
    
    # ========== 根据不同情况选择不同的分块策略 ==========
    if need_send_chunking and not need_recv_chunking:
        # 情况 1：只需要发送端分块
        # 使用完整 buf + rdispls 指定偏移
        use_recv_slice = False
        safe_chunk_size = min(max_chunk_size, MPI_DISP_LIMIT // (procs_in_my_group - 1))
        num_chunks = (local_data_size_bytes + safe_chunk_size - 1) // safe_chunk_size
        use_chunking = True
    elif need_recv_chunking:
        # 情况 2：需要接收端分块
        # 使用 buf 切片 + rdispls=0
        use_recv_slice = True
        if need_send_chunking:
            # 同时需要发送端分块：取两者最小值
            send_limit = MPI_DISP_LIMIT // (procs_in_my_group - 1)
            recv_limit = MPI_DISP_LIMIT
            safe_chunk_size = min(max_chunk_size, send_limit, recv_limit)
        else:
            # 只需要接收端分块
            safe_chunk_size = min(max_chunk_size, MPI_DISP_LIMIT)
        num_chunks = (local_data_size_bytes + safe_chunk_size - 1) // safe_chunk_size
        use_chunking = True
    else:
        # 情况 3：不需要分块
        use_recv_slice = False
        use_chunking = False
        num_chunks = 1
        safe_chunk_size = local_data_size_bytes
    # ================================================================
    
    if rank == 0:
        last_group_size = comm_size - last_group_start
        getLogger().info(f"Using {io_procs} I/O processes for {comm_size} total processes")
        getLogger().info(f"First {io_procs - 1} groups: {base_group_size} processes each")
        getLogger().info(f"Last group: {last_group_size} processes")
        getLogger().info(f"Local data size per process: {local_data_size_bytes / (1024**3):.2f} GB")
        getLogger().info(f"Max displacement in I/O buffer: {max_displacement / (1024**3):.2f} GB")
        if use_chunking:
            getLogger().info(f"Need send chunking: {need_send_chunking}, Need recv chunking: {need_recv_chunking}")
            getLogger().info(
                f"Using chunked transfer: {num_chunks} chunks "
                f"(safe chunk size: {safe_chunk_size / (1024**3):.2f} GB)"
            )
            getLogger().info(f"Using receive slice strategy: {use_recv_slice}")
    
    send_buf = None
    buf = numpy.ascontiguousarray(numpy.empty(shape, dtype=native_dtype))

    if is_io_proc:
        # Phase 1: I/O进程读取数据
        group_ranks = [my_io_leader + i for i in range(procs_in_my_group)]
        group_coords = [getCoordFromRank(r) for r in group_ranks]
        
        aggregator_shape = list(shape)
        min_coords = [min(coord[j] for coord in group_coords) for j in range(len(axes))]
        max_coords = [max(coord[j] for coord in group_coords) for j in range(len(axes))]
        
        for j, i in enumerate(axes):
            aggregator_shape[i] = shape[i] * (max_coords[j] - min_coords[j] + 1)
        
        io_buf = numpy.empty(tuple(aggregator_shape), dtype=native_dtype)
        
        global_sizes = list(shape)
        for j, i in enumerate(axes):
            global_sizes[i] *= grid[j]
        
        block_starts = [0] * len(shape)
        for j, i in enumerate(axes):
            block_starts[i] = min_coords[j] * shape[i]

        filetype_dtype_base = dtlib.from_numpy_dtype(native_dtype)
        filetype = filetype_dtype_base.Create_subarray(global_sizes, tuple(aggregator_shape), block_starts)
        filetype.Commit()

        io_comm = comm.Split(color=0 if is_io_proc else MPI.UNDEFINED, key=rank)
        
        if io_comm != MPI.COMM_NULL:
            fh = MPI.File.Open(io_comm, filename, MPI.MODE_RDONLY)
            fh.Set_view(disp=offset, filetype=filetype)
            fh.Read_all(io_buf)
            fh.Close()
            io_comm.Free()
        
        filetype.Free()

        send_buf = numpy.ascontiguousarray(numpy.empty(procs_in_my_group * numpy.prod(shape), dtype=native_dtype))
        
        for i, target_rank in enumerate(group_ranks):
            target_coords = getCoordFromRank(target_rank)
            
            source_slice = []
            for dim in range(len(shape)):
                if dim in axes:
                    j = axes.index(dim)
                    offset_in_dim = target_coords[j] - min_coords[j]
                    source_slice.append(slice(offset_in_dim * shape[dim], (offset_in_dim + 1) * shape[dim]))
                else:
                    source_slice.append(slice(None))
            
            dest_start = i * numpy.prod(shape)
            dest_end = (i + 1) * numpy.prod(shape)
            
            send_buf[dest_start:dest_end] = io_buf[tuple(source_slice)].ravel()
    else:
        io_comm = comm.Split(color=MPI.UNDEFINED, key=rank)

    if send_buf is None:
        send_buf = numpy.empty(0, dtype=native_dtype)

    comm.Barrier()
    
    # Phase 2: 数据传输
    if use_chunking:
        # 分块传输
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * safe_chunk_size
            chunk_end = min((chunk_idx + 1) * safe_chunk_size, local_data_size_bytes)
            chunk_size = chunk_end - chunk_start

            sendcounts = numpy.zeros(comm_size, dtype=numpy.int32)
            recvcounts = numpy.zeros(comm_size, dtype=numpy.int32)
            sdispls = numpy.zeros(comm_size, dtype=numpy.int32)
            rdispls = numpy.zeros(comm_size, dtype=numpy.int32)
            sendtypes = [MPI.BYTE] * comm_size
            recvtypes = [MPI.BYTE] * comm_size

            # ========== 根据策略设置接收端参数 ==========
            recvcounts[my_io_leader] = chunk_size
            if use_recv_slice:
                # 策略 1：使用切片 + rdispls=0
                rdispls[my_io_leader] = 0
            else:
                # 策略 2：使用完整 buf + rdispls=chunk_start
                rdispls[my_io_leader] = chunk_start
            # ===========================================

            if is_io_proc:
                send_buf_chunk = numpy.empty(procs_in_my_group * chunk_size, dtype=numpy.uint8)
                
                start_target_rank = my_io_group * base_group_size if my_io_group < io_procs - 1 else last_group_start
                
                for i in range(procs_in_my_group):
                    target_rank = start_target_rank + i
                    
                    src_start = i * local_data_size_bytes + chunk_start
                    src_end = i * local_data_size_bytes + chunk_start + chunk_size
                    
                    dst_start = i * chunk_size
                    dst_end = (i + 1) * chunk_size
                    
                    send_buf_chunk[dst_start:dst_end] = send_buf.view(numpy.uint8)[src_start:src_end]
                    
                    sendcounts[target_rank] = chunk_size
                    sdispls[target_rank] = i * chunk_size
            else:
                send_buf_chunk = numpy.empty(0, dtype=numpy.uint8)

            if chunk_idx == 0 and is_io_proc and rank % 16 == 0:
                max_sdispl = max(sdispls) if is_io_proc else 0
                getLogger().info(
                    f"I/O rank {rank}: chunk {chunk_idx+1}/{num_chunks}, "
                    f"chunk_size={chunk_size/(1024**2):.1f}MB, "
                    f"max_sdispl={max_sdispl/(1024**3):.2f}GB"
                )

            # ========== 根据策略选择接收 buffer ==========
            if use_recv_slice:
                # 策略 1：使用切片
                buf_chunk = buf.view(numpy.uint8)[chunk_start:chunk_end]
                comm.Alltoallw(
                    [send_buf_chunk, sendcounts, sdispls, sendtypes],
                    [buf_chunk, recvcounts, rdispls, recvtypes]
                )
            else:
                # 策略 2：使用完整 buf
                comm.Alltoallw(
                    [send_buf_chunk, sendcounts, sdispls, sendtypes],
                    [buf, recvcounts, rdispls, recvtypes]
                )
            # ===========================================
    else:
        # 不需要分块
        sendcounts = numpy.zeros(comm_size, dtype=numpy.int32)
        recvcounts = numpy.zeros(comm_size, dtype=numpy.int32)
        sdispls = numpy.zeros(comm_size, dtype=numpy.int32)
        rdispls = numpy.zeros(comm_size, dtype=numpy.int32)
        sendtypes = [MPI.BYTE] * comm_size
        recvtypes = [MPI.BYTE] * comm_size
        
        recvcounts[my_io_leader] = local_data_size_bytes

        if is_io_proc:
            start_target_rank = my_io_group * base_group_size if my_io_group < io_procs - 1 else last_group_start
            for i in range(procs_in_my_group):
                target_rank = start_target_rank + i
                sendcounts[target_rank] = local_data_size_bytes
                sdispls[target_rank] = i * local_data_size_bytes
                
            if rank % 16 == 0:
                max_displ = numpy.max(sdispls[sdispls > 0]) if numpy.any(sdispls > 0) else 0
                getLogger().info(
                    f"I/O rank {rank}: group {my_io_group}, "
                    f"procs={procs_in_my_group}, "
                    f"local_data={local_data_size_bytes/(1024**2):.1f}MB, "
                    f"max_displ={max_displ/(1024**3):.2f}GB"
                )
        
        comm.Alltoallw(
            [send_buf, sendcounts, sdispls, sendtypes],
            [buf.view(numpy.uint8), recvcounts, rdispls, recvtypes]
        )

    return buf.view(numpy_dtype)


def readMPIFileInChunks(
    filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int]
) -> Generator[Tuple[int, NDArray], None, None]:
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_RDONLY)
    filetype.Commit()
    for i in range(count):
        fh.Set_view(disp=offset + i * _MPI_SIZE * filetype.size, filetype=filetype)
        fh.Read_all(buf)
        yield i, buf.view(dtype)
    filetype.Free()
    fh.Close()


def writeMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray):
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = buf.view(native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Write_all(buf)
    filetype.Free()
    fh.Close()


def writeMPIFileInChunks(
    filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray
):
    native_dtype_str, filetype = getSubarray(dtype, shape, axes)
    buf = buf.view(native_dtype_str)

    fh = MPI.File.Open(getMPIComm(), filename, MPI.MODE_WRONLY | MPI.MODE_CREATE)
    filetype.Commit()
    for i in range(count):
        fh.Set_view(disp=offset + i * _MPI_SIZE * filetype.size, filetype=filetype)
        yield i  # Waiting for buf
        fh.Write_all(buf)
    filetype.Free()
    fh.Close()
