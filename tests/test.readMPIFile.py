from typing import List, Sequence
import sys
import numpy
from numpy.typing import NDArray, DTypeLike
from mpi4py import MPI
from mpi4py.util import dtlib

from pyquda import init
from pyquda.field import LatticeInfo

# ===================================
# Global MPI and Lattice Setup
# ===================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Nd, Ns, Nc = 4, 4, 3

# The grid is passed as command-line arguments, e.g.: mpirun -np 512 python test.readMPIFile.py 4 4 4 8
grid = [int(d) for d in sys.argv[1:]]

init(grid, [144, 144, 144, 288], backend="numpy", resource_path=".cache")
latt_info = LatticeInfo([144, 144, 144, 288], 1, 1)


# ======================================================================
# Implementation 1: Collective I/O using Set_view and Read_all
# (This is the original, high-performance version that has a bug at scale)
# ======================================================================

def _defaultCoordFromRank(rank: int, dims: Sequence[int]) -> List[int]:
    """Helper function for getSubarray.
    Calculates process coordinates from its rank.
    """
    coords = []
    for dim in dims[::-1]:
        coords.append(rank % dim)
        rank = rank // dim
    return coords[::-1]

def _getSubarray(dtype: DTypeLike, shape: Sequence[int], axes: Sequence[int]):
    """Helper function for readMPIFile_collective.
    Creates an MPI subarray datatype to describe the data layout for Set_view.
    """
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
    """Original read function using collective I/O.
    """
    native_dtype_str, filetype = _getSubarray(dtype, shape, axes)
    buf = numpy.empty(shape, native_dtype_str)

    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
    filetype.Commit()
    fh.Set_view(disp=offset, filetype=filetype)
    fh.Read_all(buf)
    filetype.Free()
    fh.Close()

    return buf.view(dtype)


# ==========================================================================
# Implementation 2: Independent I/O using Seek and Read
# (This is the workaround for the bug in the collective I/O implementation)
# ==========================================================================

def readMPIFile_independent(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray:
    """Reads a file in parallel using independent I/O operations (Seek and Read).
    """
    # This function uses the global comm and rank variables defined at the top of the script.
    dtype_info = numpy.dtype(dtype)
    native_dtype_str = dtype_info.str.replace('>', '<') if dtype_info.str.startswith('>') else dtype_info.str
    buf = numpy.empty(shape, dtype=native_dtype_str)

    local_chunk_size = buf.nbytes
    process_offset = offset + rank * local_chunk_size

    fh = MPI.File.Open(comm, filename, MPI.MODE_RDONLY)
    fh.Seek(process_offset)
    fh.Read(buf)
    fh.Close()

    return buf.view(dtype)


# ===================================
# Main Execution Block
# ===================================
if __name__ == "__main__":
    Lx, Ly, Lz, Lt = latt_info.size

    comm.Barrier()
    if rank == 0:
        print(f"Starting file read with {comm.Get_size()} processes...")

    start_time = MPI.Wtime()

    # --- Choose which implementation to run ---
    # Use the new, robust independent read:
    gauge = readMPIFile_independent(
        "/public/home/ybyang_1/data/a045m130/l144288f211b700m000569m01555m1827a.scidac.1014.01_hyp",
        ">f8",
        0,
        (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
        (7, 6, 5, 4)
    )

    # To test the original collective read, comment out the line above and uncomment the line below:
    # gauge = readMPIFile_collective(
    #     "/public/home/ybyang_1/data/a045m130/l144288f211b700m000569m01555m1827a.scidac.1014.01_hyp",
    #     ">f8",
    #     0,
    #     (Nd, Nc, Nc, 2, Lt, Lz, Ly, Lx),
    #     (7, 6, 5, 4)
    # )

    end_time = MPI.Wtime()

    if rank == 0:
        duration = end_time - start_time
        print(f"--- File read execution time: {duration:.4f} seconds ---")