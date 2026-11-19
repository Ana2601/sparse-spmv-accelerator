`timescale 1ns/1ps
// Loads one CSR matrix + vector from +in, runs y = A x, writes y and the
// performance counters to +out.
// Input file: rows nnz cols / row_ptr[rows+1] / col_idx[nnz] / val[nnz] / x[cols]
module tb_spmv;
    parameter integer P = 4;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg         wr_en = 0, start = 0;
    reg  [1:0]  wr_sel = 0;
    reg  [15:0] wr_addr = 0;
    reg  [31:0] wr_data = 0;
    reg  [10:0] n_rows = 0;
    reg  [9:0]  rd_addr = 0;
    wire        busy, done;
    wire [31:0] perf_cycles, perf_macs;
    wire signed [47:0] rd_y;

    spmv_csr #(.P(P)) dut (.*);

    integer fi, fo, r, i, rows, nnz, cols, v;
    string in_file, out_file;

    task automatic wr(input [1:0] sel, input integer addr, input integer data);
        begin
            wr_en <= 1; wr_sel <= sel; wr_addr <= addr; wr_data <= data;
            @(posedge clk);
            wr_en <= 0;
        end
    endtask

    initial begin
        if (!$value$plusargs("in=%s", in_file))   in_file = "build/m.txt";
        if (!$value$plusargs("out=%s", out_file)) out_file = "build/y.txt";
        fi = $fopen(in_file, "r");
        repeat (3) @(posedge clk);
        rst_n = 1;
        r = $fscanf(fi, "%d %d %d", rows, nnz, cols);
        for (i = 0; i <= rows; i = i + 1) begin r = $fscanf(fi, "%d", v); wr(0, i, v); end
        for (i = 0; i < nnz; i = i + 1)   begin r = $fscanf(fi, "%d", v); wr(1, i, v); end
        for (i = 0; i < nnz; i = i + 1)   begin r = $fscanf(fi, "%d", v); wr(2, i, v); end
        for (i = 0; i < cols; i = i + 1)  begin r = $fscanf(fi, "%d", v); wr(3, i, v); end
        n_rows <= rows; start <= 1;
        @(posedge clk); start <= 0;
        while (!done) @(posedge clk);
        fo = $fopen(out_file, "w");
        $fwrite(fo, "%0d %0d\n", perf_cycles, perf_macs);
        for (i = 0; i < rows; i = i + 1) begin
            rd_addr = i; #1;
            $fwrite(fo, "%0d\n", rd_y);
        end
        $fclose(fo);
        $finish;
    end
endmodule
