#include <stdint.h>
int32_t zigzag_encode(int32_t x) {
  return (x << 1) ^ (x >> 31);
}
