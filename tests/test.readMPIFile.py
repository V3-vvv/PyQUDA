# =================================================================================
#
# 这个脚本旨在使用MPI（通过mpi4py）以三种不同的并行I/O策略读取一个大型二进制数据文件，
# 并通过相互比较来验证其实现的正确性。
#
# 三种策略包括：
# 1. Collective I/O: 所有进程协同进行文件读取。
# 2. Independent I/O: 每个进程独立进行文件读取。
# 3. Aggregator I/O: 由一小部分“聚合器”进程负责文件读取，然后将数据分发给其他进程。
#
# =================================================================================
from typing import List, Sequence
import sys
import numpy
from numpy.typing import NDArray, DTypeLike
from mpi4py import MPI
from mpi4py.util import dtlib

from pyquda import init
from pyquda.field import LatticeInfo

# ===================================
# 全局MPI和格点信息初始化
# ===================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Nd, Ns, Nc = 4, 4, 3

# 从命令行参数解析进程网格布局，这决定了全局数据如何分布到各个进程。
grid = [int(d) for d in sys.argv[1:]]

init(grid, [24, 24, 24, 64], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([24, 24, 24, 64], 1, 1)


# ======================================================================
# 实现 1: 集体I/O (Collective I/O)
# ======================================================================

def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    # 从进程的rank计算其在多维进程网格中的坐标。
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]

def _getSubarray(dtype: DTypeLike, shape: Sequence[int], axes: Sequence[int]):
    # 这是一个核心辅助函数，它创建一个MPI子数组数据类型(subarray datatype)。
    # 这个数据类型精确地描述了当前进程负责的数据块在整个全局数组中的位置和大小。
    sizes = [d for d in shape]
    subsizes = [d for d in shape]
    starts = [d if i in axes else 0 for i, d in enumerate(shape)]
    coord = _defaultCoordFromRank(MPI.COMM_WORLD.Get_rank(), grid)
    for j, i in enumerate(axes):
        sizes[i] *= grid[j]
        starts[i] *= coord[j]

    dtype_str = numpy.dtype(dtype).str
    native_dtype_str = dtype_str if not dtype_str.startswith(">") else dtype_str.replace(">", "<")
    return native_dtype_str, dtlib.from_numpy_dtype(native_dtype_str).Create_subarray(sizes, subsizes, starts)

def readMPIFile_collective(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
    """使用MPI集体I/O (Read_all) 读取文件，通常在并行文件系统上效率最高。"""
    native_dtype_str, filetype = _getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
    filetype.Commit()
    
    # Set_view是关键步骤。它根据filetype为每个进程设置一个"文件视图"，
    # 使得后续的Read_all操作知道应该从文件的哪些（可能不连续的）位置读取数据。
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


# ==========================================================================
# 实现 2: 独立I/O (Independent I/O)
# ==========================================================================

def readMPIFile_independent(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
    """使用MPI独立I/O (Read) 读取文件，每个进程独立访问文件。"""
    native_dtype_str, filetype = _getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
    filetype.Commit()
    
    # 即使是独立I/O, Set_view依然是必要的，它能为每个进程正确定位文件指针。
    fh.Set_view(disp=offset, filetype=filetype)
    
    # Read()是一个非集体操作，每个进程根据自己的视图独立读取数据。
    fh.Read(buf)
    
    filetype.Free()
    fh.Close()

    return buf.view(dtype)

# ==========================================================================
# 实现 3: 聚合I/O (Aggregator I/O)
# ==========================================================================

def readMPIFile_aggregator(
    filename: str,
    dtype: DTypeLike,
    offset: int,
    shape: Sequence[int],
    axes: Sequence[int],
    io_procs: int
) -> NDArray:
    """
    采用两阶段聚合I/O模式。少数进程(io_procs)负责读取大块数据，然后通过Alltoallw分发。
    这种方法旨在减少同时访问文件系统的进程数量，可能在某些系统上提高性能。
    """
    comm_size = comm.Get_size()
    if comm_size % io_procs != 0:
        if rank == 0:
            print(f"Error: Total processes ({comm_size}) must be divisible by I/O processes ({io_procs}).")
        comm.Abort(1)

    numpy_dtype = numpy.dtype(dtype)
    native_dtype_str = numpy_dtype.str.replace('>', '<') if numpy_dtype.str.startswith('>') else numpy_dtype.str
    native_dtype = numpy.dtype(native_dtype_str)
    itemsize = native_dtype.itemsize

    is_io_proc = rank < io_procs
    # 将所有进程分为两组：I/O进程组和其他进程组。I/O进程组将拥有自己的通信器io_comm。
    io_comm = comm.Split(color=0 if is_io_proc else 1, key=rank)
    io_rank = io_comm.Get_rank() if is_io_proc else -1
    procs_per_io_node = comm_size // io_procs
    
    io_buf = None
    send_buf = None

    if is_io_proc:
        # ---- Phase 1: I/O聚合器从文件中读取大块数据 ----
        
        # 计算每个聚合器需要读取的"超级块"的形状。
        aggregator_shape = list(shape)
        fastest_changing_axis_in_grid = axes[-1]
        aggregator_shape[fastest_changing_axis_in_grid] *= procs_per_io_node
        
        io_buf = numpy.empty(tuple(aggregator_shape), dtype=native_dtype)
        
        # 为这个"超级块"创建正确的文件视图，以便正确定位和读取。
        global_sizes = list(shape)
        for j, i in enumerate(axes):
            global_sizes[i] *= grid[j]

        start_rank_in_block = io_rank * procs_per_io_node
        start_coords = _defaultCoordFromRank(start_rank_in_block, grid)
        
        block_starts = [0] * len(shape)
        for j, i in enumerate(axes):
            proc_dim_size = shape[i]
            block_starts[i] = start_coords[j] * proc_dim_size

        filetype_dtype_base = dtlib.from_numpy_dtype(native_dtype)
        filetype = filetype_dtype_base.Create_subarray(global_sizes, tuple(aggregator_shape), block_starts)
        filetype.Commit()

        # 仅I/O进程参与文件读取操作。
        fh = MPI.File.Open(io_comm, filename, MPI.MODE_RDONLY)
        fh.Set_view(disp=offset, filetype=filetype)
        fh.Read_all(io_buf)
        fh.Close()
        filetype.Free()

        # ---- 关键修正: 数据重排 ----
        # 从文件中读入的io_buf中，分属于不同目标进程的数据块在内存中是交错的、非连续的。
        # Alltoallw的简单位移计算要求发送缓冲区中的数据是连续排列的。
        # 因此，我们创建一个新的连续缓冲区send_buf，并手动将io_buf中的数据块按正确顺序复制进去。
        send_buf = numpy.empty(io_buf.shape, dtype=native_dtype)
        local_shape_t = shape[fastest_changing_axis_in_grid]
        
        for i in range(procs_per_io_node):
            # 定义源切片，从非连续的io_buf中提取第i个进程的数据。
            source_slice = [slice(None)] * len(aggregator_shape)
            source_slice[fastest_changing_axis_in_grid] = slice(i * local_shape_t, (i + 1) * local_shape_t)
            
            # 定义目标位置，在连续的send_buf中。
            dest_start = i * (numpy.prod(shape))
            dest_end = (i + 1) * (numpy.prod(shape))
            
            # 将提取出的数据块展平(ravel)后，放入send_buf的连续位置。
            send_buf.ravel()[dest_start:dest_end] = io_buf[tuple(source_slice)].ravel()

    # ---- Phase 2: 所有进程参与数据分发 ----
    sendcounts = numpy.zeros(comm_size, dtype=int)
    recvcounts = numpy.zeros(comm_size, dtype=int)
    sdispls = numpy.zeros(comm_size, dtype=int)
    rdispls = numpy.zeros(comm_size, dtype=int)
    sendtypes = [MPI.BYTE] * comm_size
    recvtypes = [MPI.BYTE] * comm_size
    
    my_io_source_rank = (rank // procs_per_io_node)
    local_data_size_bytes = numpy.prod(shape) * itemsize
    recvcounts[my_io_source_rank] = local_data_size_bytes

    if is_io_proc:
        start_target_rank = rank * procs_per_io_node
        for i in range(procs_per_io_node):
            target_rank = start_target_rank + i
            sendcounts[target_rank] = local_data_size_bytes
            sdispls[target_rank] = i * local_data_size_bytes
            
    buf = numpy.empty(shape, dtype=native_dtype)
    comm.Barrier()
    
    # 使用重排后的、数据连续的send_buf作为发送源，执行All-to-all通信。
    comm.Alltoallw(
        [send_buf, sendcounts, sdispls, sendtypes],
        [buf, recvcounts, rdispls, recvtypes]
    )

    return buf.view(numpy_dtype)

# ===================================
# 主执行与验证模块
# ===================================
if __name__ == "__main__":
    Lx, Ly, Lz, Lt = latt_info.size

    comm.Barrier()
    if rank == 0:
        print(f"Starting program with {comm.Get_size()} processes...")

    # --- 步骤 1: 分别执行三种实现 ---
    
    gauge_independent = readMPIFile_independent(
        "/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", ">f8", 0,
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4)
    )
    if rank == 0: print("readMPIFile_independent finished.")
    
    # 将collective I/O的结果作为正确性的基准("golden" reference)。
    gauge_collective = readMPIFile_collective(
        "/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", ">f8", 0,
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), (7, 6, 5, 4)
    )
    if rank == 0: print("readMPIFile_collective finished.")
    
    num_io_procs = 2 
    if comm.Get_size() < num_io_procs:
        num_io_procs = comm.Get_size()
    
    gauge_aggregator = readMPIFile_aggregator(
        filename="/public/ensemble/24I/rbc_conf_2464_m0.005_0.04_008545_hyp", dtype=">f8", offset=0,
        shape=(Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx), axes=(7, 6, 5, 4), io_procs=num_io_procs
    )
    if rank == 0: print(f"readMPIFile_aggregator with {num_io_procs} I/O procs finished.")
    
    comm.Barrier()

    # --- 步骤 2: 详细的验证过程 ---

    # 在每个进程上独立进行数值比较。
    comp_col_vs_ind = numpy.allclose(gauge_collective, gauge_independent)
    comp_col_vs_agg = numpy.allclose(gauge_collective, gauge_aggregator)
    
    # 如果验证失败，则准备包含详细错误信息的报告。
    errors = {}
    if not comp_col_vs_ind:
        diff_indices = numpy.where(gauge_collective != gauge_independent)
        first_diff_idx = tuple(d[0] for d in diff_indices)
        val_correct = gauge_collective[first_diff_idx]
        val_wrong = gauge_independent[first_diff_idx]
        errors['independent'] = (
            f"FAILED. First mismatch at index {first_diff_idx}: {val_correct} (correct) != {val_wrong} (wrong)"
        )

    if not comp_col_vs_agg:
        diff_indices = numpy.where(gauge_collective != gauge_aggregator)
        first_diff_idx = tuple(d[0] for d in diff_indices)
        val_correct = gauge_collective[first_diff_idx]
        val_wrong = gauge_aggregator[first_diff_idx]
        errors['aggregator'] = (
            f"FAILED. First mismatch at index {first_diff_idx}: {val_correct} (correct) != {val_wrong} (wrong)"
        )

    local_report = {
        "rank": rank,
        "comparisons": {
            "independent": comp_col_vs_ind,
            "aggregator": comp_col_vs_agg
        },
        "errors": errors
    }

    # 使用gather将所有进程的本地报告收集到rank 0进程。
    all_reports = comm.gather(local_report, root=0)

    # rank 0 进程负责打印所有进程的验证结果摘要。
    if rank == 0:
        print("\n--- Detailed Verification Results (Baseline: collective) ---")
        
        overall_success = True
        for report in all_reports:
            print(f"\n--- [Rank {report['rank']}] ---")
            
            if report['comparisons']['independent']:
                print("  - collective vs independent: PASSED")
            else:
                overall_success = False
                print(f"  - collective vs independent: {report['errors']['independent']}")

            if report['comparisons']['aggregator']:
                print("  - collective vs aggregator:  PASSED")
            else:
                overall_success = False
                print(f"  - collective vs aggregator:  {report['errors']['aggregator']}")
        
        print("\n-------------------------------------------------------------")
        if overall_success:
            print("SUMMARY: SUCCESS! All methods produced identical results on all processes.")
        else:
            print("SUMMARY: FAILURE! Mismatches were detected. See details above.")
        print("-------------------------------------------------------------\n")