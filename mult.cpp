#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#define BLOCK_SIZE 32

typedef struct {
    float* a;
    float* bT;
    float* c;
    int startRow;
    int endRow;
    int n;
} ThreadArgs;


extern "C" {
#include <immintrin.h>
}

void mult_original(float* a, float* b, float*c, int matrix_size, int thread_count) {
	for ( int i = 0; i < matrix_size; i++ ) {
		for ( int j = 0; j < matrix_size; j++ ) {
			c[i*matrix_size+j] = 0;
			for ( int k = 0; k < matrix_size; k++ ) {
				c[i*matrix_size+j] += a[i*matrix_size+k]*b[k*matrix_size+j];
			}
		}
	}
}

void mult_transpose(float* a, float* b, float*c, int matrix_size, int thread_count) {
	float* bt = (float*)malloc(sizeof(float)*matrix_size*matrix_size);
	for ( int i = 0; i < matrix_size; i++ ) {
		for ( int j = 0; j < matrix_size; j++ ) {
			bt[i*matrix_size+j] = b[j*matrix_size+i];
		}
	}

	for ( int i = 0; i < matrix_size; i++ ) {
		for ( int j = 0; j < matrix_size; j++ ) {
			c[i*matrix_size+j] = 0;
			for ( int k = 0; k < matrix_size; k++ ) {
				c[i*matrix_size+j] += a[i*matrix_size+k]*bt[j*matrix_size+k];
			}
		}
	}
	free(bt);
}


void transpose(float* src, float* dst, int n) {
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            dst[j*n + i] = src[i*n + j];
}



/// cache AVX 
void* avx_multiply_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*) arg;
    int blockSize = 32;  
    for (int ii = args->startRow; ii < args->endRow; ii += blockSize) {
        for (int jj = 0; jj < args->n; jj += blockSize) {
            for (int kk = 0; kk < args->n; kk += blockSize) {
                for (int i = ii; i < ii + blockSize && i < args->endRow; ++i) {
                    for (int j = jj; j < jj + blockSize && j < args->n; ++j) {
                        __m256 c_vec = _mm256_setzero_ps();
                        for (int k = kk; k < kk + blockSize && k < args->n; k += 8) {
                            __m256 a_vec = _mm256_loadu_ps(&args->a[i * args->n + k]);
                            __m256 b_vec = _mm256_loadu_ps(&args->bT[j * args->n + k]);
                            c_vec = _mm256_add_ps(c_vec, _mm256_mul_ps(a_vec, b_vec));
                        }
                        float* c_array = (float*)&c_vec;
                        for(int k = 0; k < 8; k++) {
                            args->c[i * args->n + j] += c_array[k];
                        }
                    }
                }
            }
        }
    }
    return NULL;
}

void avx_matrix_multiply(float* a, float* b, float* c, int n, int thread_count) {
    float* bT = (float*) malloc(sizeof(float) * n * n);
    transpose(b, bT, n);

    pthread_t* threads = new pthread_t[thread_count];
    ThreadArgs* threadArgs = new ThreadArgs[thread_count];

    for (int t = 0; t < thread_count; t++) {
        threadArgs[t].a = a;
        threadArgs[t].bT = bT;
        threadArgs[t].c = c;
        threadArgs[t].startRow = t * n / thread_count;
        threadArgs[t].endRow = (t == (thread_count - 1)) ? n : (t + 1) * n / thread_count;
        threadArgs[t].n = n;
        pthread_create(&threads[t], NULL, avx_multiply_thread, &threadArgs[t]);
    }

    for (int t = 0; t < thread_count; t++) {
        pthread_join(threads[t], NULL);
    }

    delete[] threads;
    delete[] threadArgs;

    free(bT);
}



///


void mult(float* a, float* b, float*c, int matrix_size, int thread_count) {
	avx_matrix_multiply(a,b,c,matrix_size,thread_count);
}
