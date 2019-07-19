// compile with: gcc -nostdlib -o paths paths.c


int read() {
    int x;
    int* y = &x;
    asm("movq %0, %%rsi;"
        "movq $0, %%rdi;"
        "movq $4, %%rdx;"
        "movq $0, %%rax;"
        "syscall;"
        : "=r"(y)
    );
    return x;
}

int func() {
    int a = 0;
    while (a < 80) {
        a = read();
    }

    int b = read();

    if (a < b) {
        return a;
    } else {
        return func();
    }
}

void main() {
    int a = read();
    int b = read();
    int c = read();

    if (a < b) {
        for (int i = 0; i < read(); i++) {
            if (b < c) {
                func();
            } else {
                func();
            }
        }
    } else {
        if (b > c) {
            func();
        } else {
            func();
        }
    }
}

void _start() {
    main();
    asm("mov $60,%rax; mov $0,%rdi; syscall");
}
