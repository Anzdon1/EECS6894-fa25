#include <stdint.h>
int varint_decode(const unsigned char* buf, int* out) {
  int shift = 0;
  int value = 0;
  for (int i = 0; i < 5; ++i) {
    unsigned char byte = buf[i];
    value |= (byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) { *out = value; return i+1; }
    shift += 7;
  }
  return -1; // malformed
}
