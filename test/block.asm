
block:     file format elf64-x86-64


Disassembly of section .text:

00000000000002b1 <compare>:
 2b1:	55                   	push   rbp
 2b2:	48 89 e5             	mov    rbp,rsp
 2b5:	89 7d fc             	mov    DWORD PTR [rbp-0x4],edi
 2b8:	89 75 f8             	mov    DWORD PTR [rbp-0x8],esi
 2bb:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
 2be:	3b 45 f8             	cmp    eax,DWORD PTR [rbp-0x8]
 2c1:	0f 9c c0             	setl   al
 2c4:	0f b6 c0             	movzx  eax,al
 2c7:	5d                   	pop    rbp
 2c8:	c3                   	ret    

00000000000002c9 <first>:
 2c9:	55                   	push   rbp
 2ca:	48 89 e5             	mov    rbp,rsp
 2cd:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
 2d1:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
 2d5:	c7 00 ef be ad de    	mov    DWORD PTR [rax],0xdeadbeef
 2db:	90                   	nop
 2dc:	5d                   	pop    rbp
 2dd:	c3                   	ret    

00000000000002de <second>:
 2de:	55                   	push   rbp
 2df:	48 89 e5             	mov    rbp,rsp
 2e2:	48 89 7d f8          	mov    QWORD PTR [rbp-0x8],rdi
 2e6:	48 8b 45 f8          	mov    rax,QWORD PTR [rbp-0x8]
 2ea:	c7 00 ad de ef be    	mov    DWORD PTR [rax],0xbeefdead
 2f0:	90                   	nop
 2f1:	5d                   	pop    rbp
 2f2:	c3                   	ret    

00000000000002f3 <main>:
 2f3:	55                   	push   rbp
 2f4:	48 89 e5             	mov    rbp,rsp
 2f7:	48 83 ec 10          	sub    rsp,0x10
 2fb:	c7 45 f8 0a 00 00 00 	mov    DWORD PTR [rbp-0x8],0xa
 302:	83 7d f8 04          	cmp    DWORD PTR [rbp-0x8],0x4
 306:	7f 09                	jg     311 <main+0x1e>
 308:	c7 45 fc 0f 00 00 00 	mov    DWORD PTR [rbp-0x4],0xf
 30f:	eb 07                	jmp    318 <main+0x25>
 311:	c7 45 fc 05 00 00 00 	mov    DWORD PTR [rbp-0x4],0x5
 318:	8b 55 fc             	mov    edx,DWORD PTR [rbp-0x4]
 31b:	8b 45 f8             	mov    eax,DWORD PTR [rbp-0x8]
 31e:	89 d6                	mov    esi,edx
 320:	89 c7                	mov    edi,eax
 322:	e8 8a ff ff ff       	call   2b1 <compare>
 327:	85 c0                	test   eax,eax
 329:	74 0e                	je     339 <main+0x46>
 32b:	48 8d 45 f4          	lea    rax,[rbp-0xc]
 32f:	48 89 c7             	mov    rdi,rax
 332:	e8 92 ff ff ff       	call   2c9 <first>
 337:	eb 0c                	jmp    345 <main+0x52>
 339:	48 8d 45 f4          	lea    rax,[rbp-0xc]
 33d:	48 89 c7             	mov    rdi,rax
 340:	e8 99 ff ff ff       	call   2de <second>
 345:	90                   	nop
 346:	c9                   	leave  
 347:	c3                   	ret    

0000000000000348 <_start>:
 348:	55                   	push   rbp
 349:	48 89 e5             	mov    rbp,rsp
 34c:	b8 00 00 00 00       	mov    eax,0x0
 351:	e8 9d ff ff ff       	call   2f3 <main>
 356:	48 c7 c0 3c 00 00 00 	mov    rax,0x3c
 35d:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 364:	0f 05                	syscall 
 366:	90                   	nop
 367:	5d                   	pop    rbp
 368:	c3                   	ret    
