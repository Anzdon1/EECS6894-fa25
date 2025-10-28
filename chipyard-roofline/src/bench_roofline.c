#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

static inline uint64_t rdcycle(void) {
  uint64_t c;
  asm volatile ("rdcycle %0" : "=r"(c));
  return c;
}

#ifndef FREQ_HZ
#define FREQ_HZ 1000000000ULL
#endif

static inline int32_t k_ops(int32_t x, int K) {
  volatile int32_t acc = x;
  for (int i = 0; i < K; ++i) {
    acc += 1;
  }
  return acc;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <N> <K>\n", argv[0]);
    return 2;
  }

  const int N = atoi(argv[1]);
  const int K = atoi(argv[2]);
  const unsigned long long bytes_in = (unsigned long long)N * 4ULL;
  const unsigned long long bytes_out = (unsigned long long)N * 4ULL;
  const unsigned long long ops = (unsigned long long)N * (unsigned long long)K;

  int32_t* src = (int32_t*)malloc(bytes_in);
  int32_t* dst = (int32_t*)malloc(bytes_out);
  if (!src || !dst) {
    fprintf(stderr, "malloc failed\n");
    return 3;
  }

  for (int i = 0; i < N; ++i) {
    src[i] = i;
  }

  const uint64_t start = rdcycle();
  for (int i = 0; i < N; ++i) {
    int32_t v = src[i];
    v = k_ops(v, K);
    dst[i] = v;
  }
  const uint64_t end = rdcycle();

  const double wall = (double)(end - start) / (double)FREQ_HZ;

  printf("BYTES_IN=%llu\n", bytes_in);
  printf("BYTES_OUT=%llu\n", bytes_out);
  printf("OPS=%llu\n", ops);
  printf("WALL=%.9f\n", wall);

  volatile int32_t guard = dst[N - 1];
  (void)guard;
  free(src);
  free(dst);
  return 0;
}
