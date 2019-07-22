void helper() {}

void func() {
    helper();
}

void main() {
    while (1) {
        func();
    }
}

void _start() {
    main();
    asm("mov $60,%rax; mov $0,%rdi; syscall");
}
