#define LIB_IO
#include "lib.h"

void main() {
    char buf[1024];

    unsigned char x = read_one_byte();
    unsigned char secret = read_one_byte();

    buf[x] = secret;

    if (x > 128) {
        x += read_one_byte();
    }

    char s = buf[x];

    write_one_byte(s);
}
