all: block-1 block-2 read loop recurse merge

%: %.c
	gcc -nostdlib -o ../target/$@ $<
	objdump -M intel -d ../target/$@ > ../target/$@.asm
