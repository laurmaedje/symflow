target := ../target
names := block-1 block-2 case twice loop recursive-1 recursive-2 func bufs
bins := $(addprefix $(target)/bin/, $(names))

all: $(bins)

$(target)/bin/%: %.c
	@mkdir -p $(target)/bin
	@mkdir -p $(target)/disassembly
	gcc -nostdlib -o $@ $<
	@objdump -M intel -d $@ > $(dir $@)../disassembly/$(notdir $@).asm

.PHONY: all
