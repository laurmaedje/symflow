.section .text
    .global _start
    .intel_syntax noprefix

_start:
    mov    rax, 0x3
    mov    qword ptr [rbp-0x8], 0x4
    add    rax, qword ptr [rbp-0x8]
    mov    qword ptr [rbp-0x4], rax

    mov    rax, 0x3c
    mov    rdi, 0x0
    syscall
