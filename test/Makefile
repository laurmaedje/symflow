all: block-1 block-2 paths read

%: %.c
	gcc -nostdlib -o ../target/$@ $<
	objdump -M intel -d ../target/$@ > ../target/$@.asm
