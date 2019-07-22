#include "lib.h"

void helper() {}

void func() {
    helper();
}

void main() {
    while (1) {
        func();
    }
}
