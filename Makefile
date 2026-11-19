RTL   := rtl/spmv_csr.sv
BUILD := build
P     ?= 4

.PHONY: test lint synth clean

test:
	@mkdir -p $(BUILD)
	iverilog -g2012 -P tb_spmv.P=$(P) -o $(BUILD)/tb.vvp $(RTL) tb/tb_spmv.sv
	python3 scripts/run_checks.py $(BUILD) $(BUILD)/tb.vvp $(P)

lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC --top-module spmv_csr $(RTL)

clean:
	rm -rf $(BUILD)
