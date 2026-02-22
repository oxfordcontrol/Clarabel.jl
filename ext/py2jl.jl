function cupy_to_cuvector(T::Type, ptr, rows)
    cu_ptr = CUDA.CuPtr{T}(pyconvert(UInt, ptr))
    return CUDA.unsafe_wrap(CUDA.CuVector{T}, cu_ptr, rows)
end
     
function cupy_to_cucsrmat(T::Type, data_ptr, indices_ptr, indptr_ptr, n_rows, n_cols, nnz)

    n_rows = pyconvert(Int32, n_rows)
    n_cols = pyconvert(Int32, n_cols)
    # Convert CuPy pointers (as integers) to Julia CUDA pointers
    data_devptr    = CUDA.CuPtr{T}(pyconvert(UInt, data_ptr))
    indices_devptr = CUDA.CuPtr{Int32}(pyconvert(UInt, indices_ptr))
    indptr_devptr  = CUDA.CuPtr{Int32}(pyconvert(UInt, indptr_ptr))

    # Wrap the raw pointers into CuArrays
    data    = unsafe_wrap(CuVector{T}, data_devptr, nnz)
    indices = unsafe_wrap(CuVector{Int32}, indices_devptr, nnz)
    indptr  = unsafe_wrap(CuVector{Int32}, indptr_devptr, n_rows + 1)

    # Index shift in Julia
    @. indices += 1
    @. indptr += 1

    # Construct the CuSparseMatrixCSR
    return CuSparseMatrixCSR(indptr, indices, data, (n_rows, n_cols))
end