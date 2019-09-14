#define LIB_IO
#include "lib.h"

void main() {
    char buf[1024];

    unsigned char x = read_one_byte();
    unsigned char y = read_one_byte();
    char secret = read_one_byte();

    buf[x] = secret;

    if (y > 128) {
        buf[y] = 'N';
    }

    char s = buf[x];

    write_one_byte(s);
}
