#define LIB_IO
#include "lib.h"

char read_one_secret_byte() {
    char secret = read_one_byte();
    return secret;
}

void main() {
    char buf[1024];

    unsigned char x = read_one_byte();
    unsigned char y = read_one_byte();
    unsigned char secret = read_one_secret_byte();

    char* a = buf + x;
    char* b = buf + 64 + y;

    a[x] = secret;
    char s = b[x];

    write_one_byte(s);
}

// Data dependency solution:
// write(a, x) -> read(b, y) <=> x = y + 64
