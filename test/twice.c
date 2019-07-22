#include "lib.h"

void helper() {}

void func() {
    helper();
}

void main() {
    func();
    func();
}

