#include "lib.h"

int fac(int n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * fac(n - 1);
    }
}

void main() {
    int x = fac(10);
}
