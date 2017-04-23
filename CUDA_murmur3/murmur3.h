#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>
#define hash_size 32

#define ROTL32(value, shift) (value << shift | value >> (32-shift))
#define BIG_CONSTANT(x) (x)
#define getblock32(p, i) (p[i])

typedef unsigned int uint;
typedef unsigned char byte;

__device__ inline uint fmix32(uint h)
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;

	return h;
}

__device__ void to_hex(uint value, char* str)
{
	char hex;
	for (int i = 0; i < 8; ++i)
	{
		hex = value % 16;
		if (hex < 10)
			str[i] = '0' + hex;
		else
			str[i] = 'a' + hex - 10;
		value = value >> 4;
	}
	//swap	
	hex = str[0]; str[0] = str[7]; str[7] = hex;
	hex = str[1]; str[1] = str[6]; str[6] = hex;
	hex = str[2]; str[2] = str[5]; str[5] = hex;
	hex = str[3]; str[3] = str[4]; str[4] = hex;
}

__device__ void MurmurHash3_x86_128(const void * key, const int len, uint seed, void * out)
{
	const byte * data = (const byte*)key;
	const int nblocks = len / 16;

	uint h1 = seed;
	uint h2 = seed;
	uint h3 = seed;
	uint h4 = seed;

	const uint c1 = 0x239b961b;
	const uint c2 = 0xab0e9789;
	const uint c3 = 0x38b34ae5;
	const uint c4 = 0xa1e38b93;

	//----------
	// body

	const uint * blocks = (const uint *)(data + nblocks * 16);

	for (int i = -nblocks; i; i++)
	{
		uint k1 = getblock32(blocks, i * 4 + 0);
		uint k2 = getblock32(blocks, i * 4 + 1);
		uint k3 = getblock32(blocks, i * 4 + 2);
		uint k4 = getblock32(blocks, i * 4 + 3);

		k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2; h1 ^= k1;

		h1 = ROTL32(h1, 19); h1 += h2; h1 = h1 * 5 + 0x561ccd1b;

		k2 *= c2; k2 = ROTL32(k2, 16); k2 *= c3; h2 ^= k2;

		h2 = ROTL32(h2, 17); h2 += h3; h2 = h2 * 5 + 0x0bcaa747;

		k3 *= c3; k3 = ROTL32(k3, 17); k3 *= c4; h3 ^= k3;

		h3 = ROTL32(h3, 15); h3 += h4; h3 = h3 * 5 + 0x96cd1c35;

		k4 *= c4; k4 = ROTL32(k4, 18); k4 *= c1; h4 ^= k4;

		h4 = ROTL32(h4, 13); h4 += h1; h4 = h4 * 5 + 0x32ac3b17;
	}

	//----------
	// tail

	const byte * tail = (const byte*)(data + nblocks * 16);

	uint k1 = 0;
	uint k2 = 0;
	uint k3 = 0;
	uint k4 = 0;

	switch (len & 15)
	{
	case 15: k4 ^= tail[14] << 16;
	case 14: k4 ^= tail[13] << 8;
	case 13: k4 ^= tail[12] << 0;
		k4 *= c4; k4 = ROTL32(k4, 18); k4 *= c1; h4 ^= k4;

	case 12: k3 ^= tail[11] << 24;
	case 11: k3 ^= tail[10] << 16;
	case 10: k3 ^= tail[9] << 8;
	case  9: k3 ^= tail[8] << 0;
		k3 *= c3; k3 = ROTL32(k3, 17); k3 *= c4; h3 ^= k3;

	case  8: k2 ^= tail[7] << 24;
	case  7: k2 ^= tail[6] << 16;
	case  6: k2 ^= tail[5] << 8;
	case  5: k2 ^= tail[4] << 0;
		k2 *= c2; k2 = ROTL32(k2, 16); k2 *= c3; h2 ^= k2;

	case  4: k1 ^= tail[3] << 24;
	case  3: k1 ^= tail[2] << 16;
	case  2: k1 ^= tail[1] << 8;
	case  1: k1 ^= tail[0] << 0;
		k1 *= c1; k1 = ROTL32(k1, 15); k1 *= c2; h1 ^= k1;
	};

	//----------
	// finalization

	h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;

	h1 += h2; h1 += h3; h1 += h4;
	h2 += h1; h3 += h1; h4 += h1;

	h1 = fmix32(h1);
	h2 = fmix32(h2);
	h3 = fmix32(h3);
	h4 = fmix32(h4);

	h1 += h2; h1 += h3; h1 += h4;
	h2 += h1; h3 += h1; h4 += h1;

	((uint*)out)[0] = h1;
	((uint*)out)[1] = h2;
	((uint*)out)[2] = h3;
	((uint*)out)[3] = h4;
}

__device__ bool compare(int N, char s[hash_size + 1])
{
	uint res[4];
	char num[10];	// N в char
	char str[33];	// hash числа N

	int len = 0;
	while (N)
	{
		num[len] = N % 10 + '0';
		N /= 10;
		++len;
	}
	num[len] = '\0';
	for (int i = 0; i < len / 2; ++i)
	{
		char c = num[i]; num[i] = num[len - i - 1]; num[len - i - 1] = c;
	}

	MurmurHash3_x86_128(num, len, 0, res);

	for (int i = 0; i < 4; ++i) to_hex(res[i], str + i * 8);
	//sprintf(str, "%lx%lx%lx%lx", res[0], res[1], res[2], res[3]);	

	for (int i = 0; i < 33; i++)
	{
		if (str[i] != s[i])
			return false;
	}
	return true;
}

__global__ void kernel(uint *data, char* hash, uint *password)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (compare(data[idx], hash))
		*password = data[idx];
}
