
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "utilities_mem.h"

void get_datatypes(std::vector<cudaDataType> *types, cudaDataType input_datatype){
	types->clear();
	types->resize(5);
	if(input_datatype==CUDA_R_32F) { // float
		types->operator[](0) = CUDA_R_32F;
		types->operator[](1) = CUDA_R_32F;
		types->operator[](2) = CUDA_R_32F;
		types->operator[](3) = CUDA_R_32F;
		types->operator[](4) = CUDA_R_32F;
	}
	else if(input_datatype==CUDA_R_64F) { // double
		types->operator[](0) = CUDA_R_64F;
		types->operator[](1) = CUDA_R_64F;
		types->operator[](2) = CUDA_R_64F;
		types->operator[](3) = CUDA_R_64F;
		types->operator[](4) = CUDA_R_64F;
	}
	else if(input_datatype==CUDA_C_32F) { // complex float
		types->operator[](0) = CUDA_C_32F;
		types->operator[](1) = CUDA_R_32F;
		types->operator[](2) = CUDA_C_32F;
		types->operator[](3) = CUDA_C_32F;
		types->operator[](4) = CUDA_C_32F;
	}
	else if(input_datatype==CUDA_C_64F) { // complex double
		types->operator[](0) = CUDA_C_64F;
		types->operator[](1) = CUDA_R_64F;
		types->operator[](2) = CUDA_C_64F;
		types->operator[](3) = CUDA_C_64F;
		types->operator[](4) = CUDA_C_64F;
	}
}

void print_matrix(int m, int n, float *A, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%e ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

void print_matrix(int m, int n, double *A, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%e ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

void check_cuSOLVER_error(cusolverStatus_t err){
	switch(err){
		case CUSOLVER_STATUS_NOT_INITIALIZED: printf("cuSOLVER error: CUSOLVER_STATUS_NOT_INITIALIZED;\n"); break;
		case CUSOLVER_STATUS_ALLOC_FAILED: printf("cuSOLVER error: CUSOLVER_STATUS_ALLOC_FAILED;\n"); break;
		case CUSOLVER_STATUS_INVALID_VALUE: printf("cuSOLVER error: CUSOLVER_STATUS_INVALID_VALUE;\n"); break;
		case CUSOLVER_STATUS_ARCH_MISMATCH: printf("cuSOLVER error: CUSOLVER_STATUS_ARCH_MISMATCH;\n"); break;
		case CUSOLVER_STATUS_EXECUTION_FAILED: printf("cuSOLVER error: CUSOLVER_STATUS_EXECUTION_FAILED;\n"); break;
		case CUSOLVER_STATUS_INTERNAL_ERROR: printf("cuSOLVER error: CUSOLVER_STATUS_INTERNAL_ERROR;\n"); break;
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: printf("cuSOLVER error: CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;\n"); break;
		default: printf("cuSOLVER error: UNDOCUMENTED WEIRD ERROR;\n");
	}
}

template<typename input_type>
void random_svd(
		input_type *A, input_type *S, input_type *U, input_type *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t k, int64_t p, int64_t nIterations, std::vector<cudaDataType> types
	){
	cusolverDnHandle_t cuSOLVER_handle = NULL;
	cudaStream_t stream = NULL;
	cusolverDnParams_t cuSOLVER_params = NULL;
	cusolverStatus_t cuSOLVER_error;
	cudaError_t CUDA_error;
	signed char jobu = 'S';
	signed char jobv = 'S';
	
	// Initiate cuSOLVER
	cuSOLVER_error = cusolverDnCreate(&cuSOLVER_handle);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) printf("Handle cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
	cuSOLVER_error = cusolverDnCreateParams(&cuSOLVER_params);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) printf("Parameter cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
	//-------------------->
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cuSOLVER_handle, stream);
	
	size_t workarea_size_device = 0;
	size_t workarea_size_host = 0;
	
	
	cuSOLVER_error = cusolverDnXgesvdr_bufferSize (
		cuSOLVER_handle, // cuSOLVER handle
		cuSOLVER_params, // cuSOLVER parameters
		jobu, // Specifies options for computing all or part of the matrix U: = 'S': the first k columns of U (the left singular vectors) are returned in the array U; = 'N': no columns of U (no left singular vectors) are computed.
		jobv, // Specifies options for computing all or part of the matrix V: = 'S': the first k rows of V (the right singular vectors) are returned in the array V; = 'N': no rows of V (no right singular vectors) are computed.
		A_nRows, // Number of rows of matrix A
		A_nCols, // Number of columns of matrix A
		k, // Rank of the k-SVD decomposition of matrix A. rank is less than min(m,n).
		p, // Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n).
		nIterations, // Number of iteration of power method.
		types[0], // Data type of array A.
		(void *) A, // Input matrix. Array of dimension lda*n with lda is not less than max(1,m). On exit, the contents of A are destroyed.
		ld_A, // Leading dimension of two-dimensional array used to store matrix A.
		types[1], // Data type of array S.
		(void *) S, // Real array of dimension min(m,n). The singular values of A, sorted so that S(i) >= S(i+1).
		types[2], // Data type of array U.
		(void *) U, // Array of dimension ldu*m with ldu is not less than max(1,m). U contains the m×m unitary matrix U. if jobu=S, only reports first min(m,n) columns of U.
		ld_U, // Leading dimension of two-dimensional array used to store matrix U.
		types[3],
		(void *) V, // Array of dimension ldv*n with ldv is not less than max(1,n). V contains the n×n unitary matrix V. If jobv=S, only reports first min(m,n) columns of V.
		ld_V, // Leading dimension of two-dimensional array used to store matrix V.
		types[4], // type
		&workarea_size_device, // Size in bytes of bufferOnDevice
		&workarea_size_host // Size in bytes of bufferOnHost
	);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) {
		check_cuSOLVER_error(cuSOLVER_error);
		printf("Buffer size cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
		printf("A:[%ld; %ld; %ld];\n", A_nRows, A_nCols, ld_A);
		printf("U:[%ld; %ld; %ld];\n", U_nRows, U_nCols, ld_U);
		printf("V:[%ld; %ld; %ld];\n", V_nRows, V_nCols, ld_V);
		printf("k = %ld; p = %ld; it = %ld;\n", k, p, nIterations);
		printf("Returned values:\n");
		printf("workarea_size_device = %zu;\n", workarea_size_device);
		printf("workarea_size_host = %zu;\n", workarea_size_host);
	}
	
	void *d_work;
	void *h_work;
	int  *d_info;
	int  h_info;
	CUDA_error = cudaMalloc((void **) &d_work, workarea_size_device);
	if(CUDA_error != cudaSuccess) printf("CUDA error %d at %s:%d\n", CUDA_error, __FILE__, __LINE__);
	h_work = (void *) malloc(workarea_size_host);
	CUDA_error = cudaMalloc((void **) &d_info, sizeof(int));
	if(CUDA_error != cudaSuccess) printf("CUDA error %d at %s:%d\n", CUDA_error, __FILE__, __LINE__);
	
	
	cuSOLVER_error = cusolverDnXgesvdr(
		cuSOLVER_handle,
		cuSOLVER_params,
		jobu, jobv,
		A_nRows, A_nCols, k, p, nIterations,
		types[0], A, ld_A, 
		types[1], S,
		types[2], U, ld_U, 
		types[3], V, ld_V,
		types[4], 
		d_work, workarea_size_device, h_work, workarea_size_host, d_info
	);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) {
		check_cuSOLVER_error(cuSOLVER_error);
		printf("SVDR cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
	}
	
	h_info = 0;
	cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
	if(h_info!=0) printf("---> h_info = %d;\n", h_info);
	
	cudaStreamSynchronize(stream);
	
	// Free cuSOLVER
	cuSOLVER_error = cusolverDnDestroyParams(cuSOLVER_params);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) printf("cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
	cuSOLVER_error = cusolverDnDestroy(cuSOLVER_handle);
	if (cuSOLVER_error != CUSOLVER_STATUS_SUCCESS) printf("cusolver error %d at %s:%d\n", cuSOLVER_error, __FILE__, __LINE__);
	
	cudaStreamDestroy(stream);
	
	free(h_work);
	cudaFree(d_work);
	cudaFree(d_info);
}

template<typename input_type>
void rSVD_GPU_wrapper(
		input_type *A, input_type *S, input_type *U, input_type *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t S_nCols, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t rank, int64_t oversampling, int64_t nIterations,
		std::vector<cudaDataType> types
	){
	// Calculate data sizes required 
	const int64_t A_size_bytes = A_nRows*A_nCols*sizeof(input_type);
	const int64_t U_size_bytes = U_nRows*U_nCols*sizeof(input_type);
	const int64_t V_size_bytes = V_nRows*V_nCols*sizeof(input_type);
	const int64_t S_size_bytes = S_nCols*sizeof(input_type);
	
	// Declare and allocate resourses on the device
	input_type *d_A = nullptr;
	input_type *d_U = nullptr;
	input_type *d_S = nullptr;
	input_type *d_V = nullptr;
	cudaMalloc((void **) &d_A, A_size_bytes);
	cudaMalloc((void **) &d_U, U_size_bytes);
	cudaMalloc((void **) &d_V, V_size_bytes);
	cudaMalloc((void **) &d_S, S_size_bytes);
	
	// Copy A matrix to the device
	cudaMemcpy(d_A, A, A_size_bytes, cudaMemcpyHostToDevice);
	
	// Run Random SVD on the device
	random_svd(d_A, d_S, d_U, d_V, A_nRows, A_nCols, ld_A, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
	
	// Copy results from the device to the host
	cudaMemcpy(U, d_U, U_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(V, d_V, V_size_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(S, d_S, S_size_bytes, cudaMemcpyDeviceToHost);
	
	cudaFree(d_A);
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_S);
}

void randomSVD_GPU_float(float *A, float *S, float *U, float *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t S_nCols, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t rank, int64_t oversampling, int64_t nIterations
	){
	std::vector<cudaDataType> types;
	get_datatypes(&types, CUDA_R_32F);
	
	rSVD_GPU_wrapper(A, S, U, V, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
}

void randomSVD_GPU_complex_float(float2 *A, float2 *S, float2 *U, float2 *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t S_nCols, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t rank, int64_t oversampling, int64_t nIterations
	){
	std::vector<cudaDataType> types;
	get_datatypes(&types, CUDA_C_32F);
	
	rSVD_GPU_wrapper(A, S, U, V, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
}

void randomSVD_GPU_double(double *A, double *S, double *U, double *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t S_nCols, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t rank, int64_t oversampling, int64_t nIterations
	){
	std::vector<cudaDataType> types;
	get_datatypes(&types, CUDA_R_64F);
	
	rSVD_GPU_wrapper(A, S, U, V, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
}

void randomSVD_GPU_complex_double(double2 *A, double2 *S, double2 *U, double2 *V, 
		int64_t A_nRows, int64_t A_nCols, int64_t ld_A, int64_t S_nCols, int64_t U_nRows, int64_t U_nCols, int64_t ld_U, int64_t V_nRows, int64_t V_nCols, int64_t ld_V, int64_t rank, int64_t oversampling, int64_t nIterations
	){
	std::vector<cudaDataType> types;
	get_datatypes(&types, CUDA_C_64F);
	
	rSVD_GPU_wrapper(A, S, U, V, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
}

void print_matrix_CM(double *A, int A_rows, int A_cols) {
	for (int row = 0; row<A_rows; row++) {
		std::cout << "(";
		std::cout << A[0*A_rows + row];
		for (int col = 1; col<A_cols; col++) {
			std::cout << ", " << A[col*A_rows + row];
		}
		std::cout << ")" << std::endl;
	}
}

void print_matrix_CM(float *A, int A_rows, int A_cols) {
	for (int row = 0; row<A_rows; row++) {
		std::cout << "(";
		std::cout << A[0*A_rows + row];
		for (int col = 1; col<A_cols; col++) {
			std::cout << ", " << A[col*A_rows + row];
		}
		std::cout << ")" << std::endl;
	}
}

void print_matrix_RM(float *A, int A_rows, int A_cols) {
	for (int row = 0; row<A_rows; row++) {
		std::cout << "(";
		std::cout << A[row*A_cols + 0];
		for (int col = 1; col<A_cols; col++) {
			std::cout << ", " << A[row*A_cols + col];
		}
		std::cout << ")" << std::endl;
	}
}

void print_matrix_raw(float *A, int nElements){
	for(int f=0; f<nElements; f++){
		std::cout << A[f] << ", ";
	}
	std::cout << std::endl;
}

void print_matrix_raw(double *A, int nElements){
	for(int f=0; f<nElements; f++){
		std::cout << A[f] << ", ";
	}
	std::cout << std::endl;
}

#ifdef __cplusplus
extern "C" {
#endif

void random_svd_python(Mem *S, Mem *U, Mem *V, Mem *A, int32_t rank, int32_t oversampling, int32_t nIterations){
	if(mem_num_dims(A) != 2 || mem_num_dims(U) != 2 || mem_num_dims(V) != 2) {
		printf("Error: Matrices have wrong dimensions.\n");
		return;
	}
	
	// Beware numpy stores data in row-major format. Here we require column-major format!
	// Use order='F' to create column major array: f = np.array([[1,2,3],[4,6,7]], order='F')
	
	int64_t A_nRows, A_nCols, ld_A;
	if(mem_is_c_contiguous(A)){
		printf("Error: NumPy array must be stored in the column-major (order='F') format\n");
		return;
	}
	else {
		// column-major format
		A_nRows = mem_shape_dim(A, 0);
		A_nCols = mem_shape_dim(A, 1);
		ld_A    = A_nRows;
	}
	
	const int64_t U_nRows = A_nRows;
	const int64_t U_nCols = rank;
	const int64_t ld_U    = U_nRows;
	const int64_t V_nRows = rank;
	const int64_t V_nCols = A_nCols;
	const int64_t ld_V    = V_nCols;
	const int64_t S_nCols = rank;
	
	if( (rank + oversampling) > min(A_nRows, A_nCols)){
		printf("Error: oversampling + rank must be <= min(A_nRows, A_nCols).\n");
		return;
	}
	
	if( mem_location(A) == MEM_CPU ) {
		if(mem_type(A) == MEM_FLOAT){
			float *A_data, *S_data, *U_data, *V_data;
			A_data = (float *) mem_data(A);
			S_data = (float *) mem_data(S);
			U_data = (float *) mem_data(U);
			V_data = (float *) mem_data(V);

			randomSVD_GPU_float(A_data, S_data, U_data, V_data, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations);
		}
		else if(mem_type(A) == MEM_COMPLEX_FLOAT){
			float2 *A_data, *S_data, *U_data, *V_data;
			A_data = (float2 *) mem_data(A);
			S_data = (float2 *) mem_data(S);
			U_data = (float2 *) mem_data(U);
			V_data = (float2 *) mem_data(V);
			
			randomSVD_GPU_complex_float(A_data, S_data, U_data, V_data, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations);
		}
		else if(mem_type(A) == MEM_DOUBLE){
			double *A_data, *S_data, *U_data, *V_data;
			A_data = (double *) mem_data(A);
			S_data = (double *) mem_data(S);
			U_data = (double *) mem_data(U);
			V_data = (double *) mem_data(V);
			
			randomSVD_GPU_double(A_data, S_data, U_data, V_data, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations);
		}
		else if(mem_type(A) == MEM_COMPLEX_DOUBLE){
			double2 *A_data, *S_data, *U_data, *V_data;
			A_data = (double2 *) mem_data(A);
			S_data = (double2 *) mem_data(S);
			U_data = (double2 *) mem_data(U);
			V_data = (double2 *) mem_data(V);
			
			randomSVD_GPU_complex_double(A_data, S_data, U_data, V_data, A_nRows, A_nCols, ld_A, S_nCols, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations);
		}
		else {
			printf("Unsupported data type.\n");
			return;
		}
	}
	else if( mem_location(A) == MEM_GPU ) {
		if(mem_type(A) == MEM_FLOAT){
			float *d_A_data, *d_S_data, *d_U_data, *d_V_data;
			d_A_data = (float *) mem_data(A);
			d_S_data = (float *) mem_data(S);
			d_U_data = (float *) mem_data(U);
			d_V_data = (float *) mem_data(V);
			
			std::vector<cudaDataType> types;
			get_datatypes(&types, CUDA_R_32F);
			
			random_svd(d_A_data, d_S_data, d_U_data, d_V_data, A_nRows, A_nCols, ld_A, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
		}
		else if(mem_type(A) == MEM_COMPLEX_FLOAT){
			float2 *d_A_data, *d_S_data, *d_U_data, *d_V_data;
			d_A_data = (float2 *) mem_data(A);
			d_S_data = (float2 *) mem_data(S);
			d_U_data = (float2 *) mem_data(U);
			d_V_data = (float2 *) mem_data(V);
			
			std::vector<cudaDataType> types;
			get_datatypes(&types, CUDA_C_32F);
			
			random_svd(d_A_data, d_S_data, d_U_data, d_V_data, A_nRows, A_nCols, ld_A, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
		}
		else if(mem_type(A) == MEM_DOUBLE){
			double *d_A_data, *d_S_data, *d_U_data, *d_V_data;
			d_A_data = (double *) mem_data(A);
			d_S_data = (double *) mem_data(S);
			d_U_data = (double *) mem_data(U);
			d_V_data = (double *) mem_data(V);
			
			std::vector<cudaDataType> types;
			get_datatypes(&types, CUDA_R_64F);
			
			random_svd(d_A_data, d_S_data, d_U_data, d_V_data, A_nRows, A_nCols, ld_A, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
		}
		else if(mem_type(A) == MEM_COMPLEX_DOUBLE){
			double2 *d_A_data, *d_S_data, *d_U_data, *d_V_data;
			d_A_data = (double2 *) mem_data(A);
			d_S_data = (double2 *) mem_data(S);
			d_U_data = (double2 *) mem_data(U);
			d_V_data = (double2 *) mem_data(V);
			
			std::vector<cudaDataType> types;
			get_datatypes(&types, CUDA_C_64F);
			
			random_svd(d_A_data, d_S_data, d_U_data, d_V_data, A_nRows, A_nCols, ld_A, U_nRows, U_nCols, ld_U, V_nRows, V_nCols, ld_V, rank, oversampling, nIterations, types);
		}
		else {
			printf("Unsupported data type.\n");
			return;
		}
	}
}

#ifdef __cplusplus
}
#endif


