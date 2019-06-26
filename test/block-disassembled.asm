
block:     file format elf64-x86-64


Disassembly of section .text:

00000000000002b1 <compare>:
 2b1:	55                   	push   %rbp
 2b2:	48 89 e5             	mov    %rsp,%rbp
 2b5:	89 7d fc             	mov    %edi,-0x4(%rbp)
 2b8:	89 75 f8             	mov    %esi,-0x8(%rbp)
 2bb:	8b 45 fc             	mov    -0x4(%rbp),%eax
 2be:	3b 45 f8             	cmp    -0x8(%rbp),%eax
 2c1:	0f 9c c0             	setl   %al
 2c4:	0f b6 c0             	movzbl %al,%eax
 2c7:	5d                   	pop    %rbp
 2c8:	c3                   	retq

00000000000002c9 <first>:
 2c9:	55                   	push   %rbp
 2ca:	48 89 e5             	mov    %rsp,%rbp
 2cd:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
 2d1:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
 2d5:	c7 00 ef be ad de    	movl   $0xdeadbeef,(%rax)
 2db:	90                   	nop
 2dc:	5d                   	pop    %rbp
 2dd:	c3                   	retq

00000000000002de <second>:
 2de:	55                   	push   %rbp
 2df:	48 89 e5             	mov    %rsp,%rbp
 2e2:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
 2e6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
 2ea:	c7 00 ad de ef be    	movl   $0xbeefdead,(%rax)
 2f0:	90                   	nop
 2f1:	5d                   	pop    %rbp
 2f2:	c3                   	retq

00000000000002f3 <main>:
 2f3:	55                   	push   %rbp
 2f4:	48 89 e5             	mov    %rsp,%rbp
 2f7:	48 83 ec 10          	sub    $0x10,%rsp
 2fb:	c7 45 f8 0a 00 00 00 	movl   $0xa,-0x8(%rbp)
 302:	83 7d f8 04          	cmpl   $0x4,-0x8(%rbp)
 306:	7f 09                	jg     311 <main+0x1e>
 308:	c7 45 fc 0f 00 00 00 	movl   $0xf,-0x4(%rbp)
 30f:	eb 07                	jmp    318 <main+0x25>
 311:	c7 45 fc 05 00 00 00 	movl   $0x5,-0x4(%rbp)
 318:	8b 55 fc             	mov    -0x4(%rbp),%edx
 31b:	8b 45 f8             	mov    -0x8(%rbp),%eax
 31e:	89 d6                	mov    %edx,%esi
 320:	89 c7                	mov    %eax,%edi
 322:	e8 8a ff ff ff       	callq  2b1 <compare>
 327:	85 c0                	test   %eax,%eax
 329:	74 0e                	je     339 <main+0x46>
 32b:	48 8d 45 f4          	lea    -0xc(%rbp),%rax
 32f:	48 89 c7             	mov    %rax,%rdi
 332:	e8 92 ff ff ff       	callq  2c9 <first>
 337:	eb 0c                	jmp    345 <main+0x52>
 339:	48 8d 45 f4          	lea    -0xc(%rbp),%rax
 33d:	48 89 c7             	mov    %rax,%rdi
 340:	e8 99 ff ff ff       	callq  2de <second>
 345:	90                   	nop
 346:	c9                   	leaveq
 347:	c3                   	retq

0000000000000348 <_start>:
 348:	55                   	push   %rbp
 349:	48 89 e5             	mov    %rsp,%rbp
 34c:	b8 00 00 00 00       	mov    $0x0,%eax
 351:	e8 9d ff ff ff       	callq  2f3 <main>
 356:	48 c7 c0 3c 00 00 00 	mov    $0x3c,%rax
 35d:	48 c7 c7 00 00 00 00 	mov    $0x0,%rdi
 364:	0f 05                	syscall
 366:	90                   	nop
 367:	5d                   	pop    %rbp
 368:	c3                   	retq
