names := \
	block-1 block-2 case twice loop func \
	recursive-1 recursive-2 \
	bufs paths deep overwrite

target := ../target
bins := $(addprefix $(target)/bin/, $(names))

all: $(bins)

$(target)/bin/%: %.c
	@mkdir -p $(target)/bin
	@mkdir -p $(target)/dis
	gcc -nostdlib -o $@ $<
	@objdump -M intel -d $@ > $(dir $@)../dis/$(notdir $@).asm

.PHONY: all
