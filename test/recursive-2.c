#include "lib.h"

void bar();

void foo() {
    bar();
}

void bar() {
    foo();
}

void main() {
    foo();
}
