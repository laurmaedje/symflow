names := \
	block-1 block-2 case twice loop func \
	recursive-1 recursive-2 \
	bufs paths deep overwrite \
	min

target := ../target
bins := $(addprefix $(target)/bin/, $(names))

define disasm
	@objdump -M intel -d $(1) > $(dir $(1))../dis/$(notdir $(1)).asm
endef

define mk
	@mkdir -p $(target)/bin
	@mkdir -p $(target)/dis
endef


all: $(bins)

$(target)/bin/%: %.c
	$(call mk)
	gcc -nostdlib -o $@ $<
	$(call disasm, $@)

$(target)/bin/%: %.s
	$(call mk)
	as -o $@ $<
	@strip $@
	$(call disasm, $@)

.PHONY: all
