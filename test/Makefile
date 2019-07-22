target := ../target/
names := block-1 block-2 case twice loop recursive-1 recursive-2 func
bins := $(addprefix $(target), $(names))

all: $(bins)

$(target)%: %.c
	gcc -nostdlib -o $@ $<
	objdump -M intel -d ../target/$@ > ../target/$@.asm

.PHONY: all
