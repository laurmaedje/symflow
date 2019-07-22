#define LIB_IO
#include "lib.h"

char left() { return 'L'; }
char right() { return 'R'; }

void main() {
    char x;
    char (*func)();

    if (read_one_byte() <= 64) {
        func = left;
    } else {
        func = right;
    }

    write_one_byte(func());
}
