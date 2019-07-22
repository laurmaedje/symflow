void helper() {}

void func() {
    helper();
}

void main() {
    func();
    func();
}

void _start() {
    main();
    asm("mov $60,%rax; mov $0,%rdi; syscall");
}
