
paths:     file format elf64-x86-64


Disassembly of section .text:

00000000000002b1 <read>:
 2b1:	55                   	push   rbp
 2b2:	48 89 e5             	mov    rbp,rsp
 2b5:	48 8d 45 f4          	lea    rax,[rbp-0xc]
 2b9:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 2bd:	48 89 c6             	mov    rsi,rax
 2c0:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 2c7:	48 c7 c2 04 00 00 00 	mov    rdx,0x4
 2ce:	48 c7 c0 00 00 00 00 	mov    rax,0x0
 2d5:	0f 05                	syscall 
 2d7:	48 89 45 f8          	mov    QWORD PTR [rbp-0x8],rax
 2db:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
 2de:	5d                   	pop    rbp
 2df:	c3                   	ret    

00000000000002e0 <func>:
 2e0:	55                   	push   rbp
 2e1:	48 89 e5             	mov    rbp,rsp
 2e4:	48 83 ec 10          	sub    rsp,0x10
 2e8:	c7 45 fc 00 00 00 00 	mov    DWORD PTR [rbp-0x4],0x0
 2ef:	eb 0d                	jmp    2fe <func+0x1e>
 2f1:	b8 00 00 00 00       	mov    eax,0x0
 2f6:	e8 b6 ff ff ff       	call   2b1 <read>
 2fb:	89 45 fc             	mov    DWORD PTR [rbp-0x4],eax
 2fe:	83 7d fc 4f          	cmp    DWORD PTR [rbp-0x4],0x4f
 302:	7e ed                	jle    2f1 <func+0x11>
 304:	b8 00 00 00 00       	mov    eax,0x0
 309:	e8 a3 ff ff ff       	call   2b1 <read>
 30e:	89 45 f8             	mov    DWORD PTR [rbp-0x8],eax
 311:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
 314:	3b 45 f8             	cmp    eax,DWORD PTR [rbp-0x8]
 317:	7d 05                	jge    31e <func+0x3e>
 319:	8b 45 fc             	mov    eax,DWORD PTR [rbp-0x4]
 31c:	eb 0a                	jmp    328 <func+0x48>
 31e:	b8 00 00 00 00       	mov    eax,0x0
 323:	e8 b8 ff ff ff       	call   2e0 <func>
 328:	c9                   	leave  
 329:	c3                   	ret    

000000000000032a <main>:
 32a:	55                   	push   rbp
 32b:	48 89 e5             	mov    rbp,rsp
 32e:	48 83 ec 10          	sub    rsp,0x10
 332:	b8 00 00 00 00       	mov    eax,0x0
 337:	e8 75 ff ff ff       	call   2b1 <read>
 33c:	89 45 f8             	mov    DWORD PTR [rbp-0x8],eax
 33f:	b8 00 00 00 00       	mov    eax,0x0
 344:	e8 68 ff ff ff       	call   2b1 <read>
 349:	89 45 f4             	mov    DWORD PTR [rbp-0xc],eax
 34c:	b8 00 00 00 00       	mov    eax,0x0
 351:	e8 5b ff ff ff       	call   2b1 <read>
 356:	89 45 f0             	mov    DWORD PTR [rbp-0x10],eax
 359:	8b 45 f8             	mov    eax,DWORD PTR [rbp-0x8]
 35c:	3b 45 f4             	cmp    eax,DWORD PTR [rbp-0xc]
 35f:	7d 3c                	jge    39d <main+0x73>
 361:	c7 45 fc 00 00 00 00 	mov    DWORD PTR [rbp-0x4],0x0
 368:	eb 22                	jmp    38c <main+0x62>
 36a:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
 36d:	3b 45 f0             	cmp    eax,DWORD PTR [rbp-0x10]
 370:	7d 0c                	jge    37e <main+0x54>
 372:	b8 00 00 00 00       	mov    eax,0x0
 377:	e8 64 ff ff ff       	call   2e0 <func>
 37c:	eb 0a                	jmp    388 <main+0x5e>
 37e:	b8 00 00 00 00       	mov    eax,0x0
 383:	e8 58 ff ff ff       	call   2e0 <func>
 388:	83 45 fc 01          	add    DWORD PTR [rbp-0x4],0x1
 38c:	b8 00 00 00 00       	mov    eax,0x0
 391:	e8 1b ff ff ff       	call   2b1 <read>
 396:	39 45 fc             	cmp    DWORD PTR [rbp-0x4],eax
 399:	7c cf                	jl     36a <main+0x40>
 39b:	eb 1e                	jmp    3bb <main+0x91>
 39d:	8b 45 f4             	mov    eax,DWORD PTR [rbp-0xc]
 3a0:	3b 45 f0             	cmp    eax,DWORD PTR [rbp-0x10]
 3a3:	7e 0c                	jle    3b1 <main+0x87>
 3a5:	b8 00 00 00 00       	mov    eax,0x0
 3aa:	e8 31 ff ff ff       	call   2e0 <func>
 3af:	eb 0a                	jmp    3bb <main+0x91>
 3b1:	b8 00 00 00 00       	mov    eax,0x0
 3b6:	e8 25 ff ff ff       	call   2e0 <func>
 3bb:	90                   	nop
 3bc:	c9                   	leave  
 3bd:	c3                   	ret    

00000000000003be <_start>:
 3be:	55                   	push   rbp
 3bf:	48 89 e5             	mov    rbp,rsp
 3c2:	b8 00 00 00 00       	mov    eax,0x0
 3c7:	e8 5e ff ff ff       	call   32a <main>
 3cc:	48 c7 c0 3c 00 00 00 	mov    rax,0x3c
 3d3:	48 c7 c7 00 00 00 00 	mov    rdi,0x0
 3da:	0f 05                	syscall 
 3dc:	90                   	nop
 3dd:	5d                   	pop    rbp
 3de:	c3                   	ret    
