
read:     file format elf64-x86-64


Disassembly of section .text:

00000000000002b1 <read_one_byte>:
 2b1:	55                   	push   rbp
 2b2:	48 89 e5             	mov    rbp,rsp
 2b5:	48 8d 45 f7          	lea    rax,[rbp-0x9]
 2b9:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 2bd:	48 89 c6             	mov    rsi,rax
 2c0:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 2c7:	48 c7 c2 01 00 00 00 	mov    rdx,0x1
 2ce:	48 c7 c0 00 00 00 00 	mov    rax,0x0
 2d5:	0f 05                	syscall 
 2d7:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 2db:	0f b6 45 f7          	movzx  eax,BYTE PTR [rbp-0x9]
 2df:	5d                   	pop    rbp
 2e0:	c3                   	ret    

00000000000002e1 <write_one_byte>:
 2e1:	55                   	push   rbp
 2e2:	48 89 e5             	mov    rbp,rsp
 2e5:	89 f8                	mov    eax,edi
 2e7:	88 45 ec             	mov    BYTE PTR [rbp-0x14],al
 2ea:	48 8d 45 ec          	lea    rax,[rbp-0x14]
 2ee:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 2f2:	48 89 c6             	mov    rsi,rax
 2f5:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 2fc:	48 c7 c2 01 00 00 00 	mov    rdx,0x1
 303:	48 c7 c0 01 00 00 00 	mov    rax,0x1
 30a:	0f 05                	syscall 
 30c:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 310:	90                   	nop
 311:	5d                   	pop    rbp
 312:	c3                   	ret    

0000000000000313 <main>:
 313:	55                   	push   rbp
 314:	48 89 e5             	mov    rbp,rsp
 317:	48 83 ec 10          	sub    rsp,0x10
 31b:	b8 00 00 00 00       	mov    eax,0x0
 320:	e8 8c ff ff ff       	call   2b1 <read_one_byte>
 325:	88 45 ff             	mov    BYTE PTR [rbp-0x1],al
 328:	80 7d ff 60          	cmp    BYTE PTR [rbp-0x1],0x60
 32c:	7e 19                	jle    347 <main+0x34>
 32e:	80 7d ff 7a          	cmp    BYTE PTR [rbp-0x1],0x7a
 332:	7f 13                	jg     347 <main+0x34>
 334:	0f b6 45 ff          	movzx  eax,BYTE PTR [rbp-0x1]
 338:	83 e8 20             	sub    eax,0x20
 33b:	0f be c0             	movsx  eax,al
 33e:	89 c7                	mov    edi,eax
 340:	e8 9c ff ff ff       	call   2e1 <write_one_byte>
 345:	eb 0b                	jmp    352 <main+0x3f>
 347:	0f be 45 ff          	movsx  eax,BYTE PTR [rbp-0x1]
 34b:	89 c7                	mov    edi,eax
 34d:	e8 8f ff ff ff       	call   2e1 <write_one_byte>
 352:	90                   	nop
 353:	c9                   	leave  
 354:	c3                   	ret    

0000000000000355 <_start>:
 355:	55                   	push   rbp
 356:	48 89 e5             	mov    rbp,rsp
 359:	b8 00 00 00 00       	mov    eax,0x0
 35e:	e8 b0 ff ff ff       	call   313 <main>
 363:	48 c7 c0 3c 00 00 00 	mov    rax,0x3c
 36a:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 371:	0f 05                	syscall 
 373:	90                   	nop
 374:	5d                   	pop    rbp
 375:	c3                   	ret    
