#include <stdio.h>
#include <ctime>
#include <locale.h>
#include "murmur3.h"

#define N		1024*1000
#define SIZE	4*N

#define default_path "C:/Users/Admin/Desktop/murmur_hash.txt"

char* read_hash(const char* path)
{
	char* hash = new char[hash_size + 1];
	FILE* file = fopen(path, "r");
	for (int i = 0; i < hash_size; ++i)
		hash[i] = getc(file);
	hash[hash_size] = '\0';
	fclose(file);
	return hash;
}

uint main(int argc, char** argv)
{
	setlocale(LC_ALL, ".1251");
	printf("Брутфорс хэша Murmur3 с использованием CUDA\n\n");
	clock_t start_time, span;

	char* hash_host = read_hash((argc > 1) ? argv[1] : default_path);
	uint pass_host = 0;
	uint *num_host = (uint *)malloc(SIZE);

	char* hash_dev;		cudaMalloc((void **)&hash_dev, hash_size + 1);
	uint* pass_dev;		cudaMalloc((void**)&pass_dev, 4);
	uint *num_dev;		cudaMalloc((void **)&num_dev, SIZE);

	start_time = clock();
	cudaMemcpy(hash_dev, hash_host, hash_size + 1, cudaMemcpyHostToDevice);
	cudaMemcpy(pass_dev, &pass_host, 4, cudaMemcpyHostToDevice);
	for (uint i = 0; i < 1000000000; i += N)
	{
		for (uint j = 0; j < N; ++j)
			num_host[j] = j + i;

		cudaMemcpy(num_dev, num_host, SIZE, cudaMemcpyHostToDevice);
		kernel << < dim3(N / 1024, 1, 1), dim3(1024, 1, 1) >> > (num_dev, hash_dev, pass_dev);
		cudaMemcpy(&pass_host, pass_dev, 4, cudaMemcpyDeviceToHost);

		if (pass_host > 0) break;
	}
	span = clock() - start_time;

	printf("Пароль: '%d'.\n", pass_host);
	printf("Время: %d s %d ms\n\n", span / 1000, span - 1000 * (span / 1000));

	free(num_host);
	cudaFree(num_dev); cudaFree(hash_dev); cudaFree(pass_dev);

	getchar();
	return 0;
}