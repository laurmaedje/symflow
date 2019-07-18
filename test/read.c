// compile with: gcc -nostdlib -o read read.c


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

void main() {
    char x = read_one_byte();

    if (x >= 'a' && x <= 'z') {
        write_one_byte(x - 32);
    } else {
        write_one_byte(x);
    }
}

void _start() {
    main();
    asm("mov $60,%rax; mov $0,%rdi; syscall");
}
