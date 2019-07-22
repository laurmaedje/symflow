#define LIB_IO
#include "lib.h"

void main() {
    char x = read_one_byte();

    if (x >= 'a' && x <= 'z') {
        write_one_byte(x - 32);
    } else {
        write_one_byte(x);
    }
}

