void main();

void _start() {
    main();
    asm("mov $60,%rax; mov $0,%rdi; syscall");
}

#ifdef LIB_IO

// Read one byte from stdin.
char read_one_byte() {
    char x;
    char* y = &x;
    asm("movq %0, %%rsi;"
        "movq $0, %%rdi;"
        "movq $1, %%rdx;"
        "movq $0, %%rax;"
        "syscall;"
        : "=r"(y)
    );
    return x;
}

// Read one byte to stdout.
void write_one_byte(char x) {
    char* y = &x;
    asm("movq %0, %%rsi;"
        "movq $0, %%rdi;"
        "movq $1, %%rdx;"
        "movq $1, %%rax;"
        "syscall;"
        : "=r"(y)
    );
}

#endif
