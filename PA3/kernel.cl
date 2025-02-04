__kernel void matrixMultiply(
		__global const int *A, __global const int *B, __global int *C,
		const unsigned int numARows, const unsigned int numAColumns,
		const unsigned int numBRows, const unsigned int numBColumns,
		const unsigned int numCRows, const unsigned int numCColumns) {

	//@@ Compute C = A^T B 

	float sum = 0;
	/*float A_row [numAColumns] = {0};*/

	int A_ind = get_global_id(0);
	int k;

	/*for(int A_loc = 0; A_loc < numAColumns; A_loc++) {*/
	/*	A_row[A_loc] = A[A_ind * numAColumns + A_loc];*/
	/*}*/

	for (int B_ind = 0; B_ind < numBColumns; B_ind++){
		sum = 0;
		for (k = 0; k < numBRows; k++){
			float a = A[k * numAColumns + A_ind];
			float b = B[k * numBColumns + B_ind];
			sum += a * b;
		}
		C[A_ind * numCColumns + B_ind] = sum;
	}
}
