# Convert a Julia CuVector to a CuPy ndarray
def JuliaCuVector2CuPyArray(jl_arr: jl_VectorValue):
    # Get the device pointer from Julia
    pDevice = jl.Int(jl.pointer(jl_arr))

    # Get array length and element type
    span = jl.size(jl_arr)
    dtype = jl.eltype(jl_arr)

    # Map Julia type to CuPy dtype
    if dtype == jl.Float64:
        dtype = cp.float64
    else:
        dtype = cp.float32

    # Compute memory size in bytes (assuming 1D vector)
    size_bytes = int(span[0] * cp.dtype(dtype).itemsize)

    # Create CuPy memory view from the Julia pointer
    mem = UnownedMemory(pDevice, size_bytes, owner=None)
    memptr = MemoryPointer(mem, 0)

    # Wrap into CuPy ndarray
    arr = cp.ndarray(shape=span, dtype=dtype, memptr=memptr)
    return arr


# Convert a CuPy ndarray to a Julia CuVector
def CuPyArray2JuliaCuVector(arr: cp.ndarray):
    ptr = arr.data.ptr
    rows = arr.shape[0]
    return jl.cupy_to_cuvector(ptr, rows)


# Convert a CuPy ndarray to a Julia CuMatrix
def CuPyArray2JuliaCuMatrix(arr: cp.ndarray):
    ptr = arr.data.ptr
    rows, cols = arr.shape
    return jl.cupy_to_cumatrix(ptr, rows, cols)



