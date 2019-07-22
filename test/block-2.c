#include "lib.h"

int compare(int a, int b) {
    return a < b;
}

void first(int* c) {
    *c = 0xdeadbeef;
}

void second(int* c) {
    *c = 0xbeefdead;
}

void main() {
    int a = 7, b, c;

    if (a < 5) {
        b = 15;
    } else {
        b = 5;
    }

    if (compare(a, b)) {
        first(&c);
    } else {
        second(&c);
    }
}
