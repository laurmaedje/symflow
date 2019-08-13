#define LIB_IO
#include "lib.h"

void main() {
    char buf[1024];

    unsigned char x = read_one_byte();
    unsigned char y = read_one_byte();
    char secret = read_one_byte();

    unsigned char* a = buf + x;
    unsigned char* b = buf + y;

    b[x] = secret;
    char s = b[a[x]];

    write_one_byte(s);
}

// Data dependency solution:
// write(b, x) -> read(a, x) <=> x = y
// write(b, x) -> read(b, a[x]) <=> x == if (x == y) then secret else mem
