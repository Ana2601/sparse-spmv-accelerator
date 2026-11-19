// Sparse matrix x dense vector (y = A x) accelerator, CSR format.
//
//  * int16 matrix values and vector entries, 48-bit accumulation (exact)
//  * P parallel MAC lanes consume P non-zeros of the current row per clock,
//    reduced by an adder tree into the row accumulator
//  * a row costs max(1, ceil(nnz_row / P)) clocks; empty rows cost 1
//  * row_ptr / col_idx / values / x live in on-chip buffers loaded through a
//    write port; y is read back through a read port
//  * performance counters: busy cycles and lane-slots used, so lane
//    utilisation = macs / (cycles * P) can be measured per matrix
module spmv_csr #(
    parameter integer P     = 4,       // MAC lanes
    parameter integer RW    = 10,      // log2(max rows)
    parameter integer NZW   = 16,      // log2(max non-zeros)
    parameter integer XW    = 10       // log2(max columns)
) (
    input  wire               clk,
    input  wire               rst_n,

    // buffer write port
    input  wire               wr_en,
    input  wire [1:0]         wr_sel,      // 0 row_ptr, 1 col_idx, 2 value, 3 x
    input  wire [NZW-1:0]     wr_addr,
    input  wire [31:0]        wr_data,

    input  wire               start,
    input  wire [RW:0]        n_rows,
    output reg                busy,
    output reg                done,
    output reg  [31:0]        perf_cycles,
    output reg  [31:0]        perf_macs,

    input  wire [RW-1:0]      rd_addr,
    output wire signed [47:0] rd_y
);
    reg [NZW:0]       row_ptr [0:(1<<RW)];
    reg [XW-1:0]      col_idx [0:(1<<NZW)-1];
    reg signed [15:0] val     [0:(1<<NZW)-1];
    reg signed [15:0] xv      [0:(1<<XW)-1];
    reg signed [47:0] y       [0:(1<<RW)-1];

    always @(posedge clk) if (wr_en && !busy) begin
        case (wr_sel)
            2'd0: row_ptr[wr_addr[RW:0]]  <= wr_data[NZW:0];
            2'd1: col_idx[wr_addr]        <= wr_data[XW-1:0];
            2'd2: val[wr_addr]            <= wr_data[15:0];
            default: xv[wr_addr[XW-1:0]]  <= wr_data[15:0];
        endcase
    end
    assign rd_y = y[rd_addr];

    // ------------------------------------------------------------ datapath
    reg [RW:0]        row;
    reg [NZW:0]       p, p_end;
    reg signed [47:0] acc;

    reg signed [47:0] lane_sum;
    reg [$clog2(P+1)-1:0] lanes_used;
    integer l;
    always @* begin
        lane_sum = 48'sd0; lanes_used = 0;
        for (l = 0; l < P; l = l + 1)
            if (p + l < p_end) begin
                lane_sum   = lane_sum + val[p + l] * xv[col_idx[p + l]];
                lanes_used = lanes_used + 1'b1;
            end
    end

    wire row_last = (p + P >= p_end);              // this clock finishes the row

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0; done <= 1'b0; row <= 0; p <= 0; p_end <= 0; acc <= 48'sd0;
            perf_cycles <= 32'd0; perf_macs <= 32'd0;
        end else begin
            done <= 1'b0;
            if (!busy && start) begin
                busy <= 1'b1; row <= 0; acc <= 48'sd0;
                p <= row_ptr[0]; p_end <= row_ptr[1];
                perf_cycles <= 32'd0; perf_macs <= 32'd0;
            end else if (busy) begin
                perf_cycles <= perf_cycles + 32'd1;
                perf_macs   <= perf_macs + lanes_used;
                if (row_last) begin
                    y[row[RW-1:0]] <= acc + lane_sum;
                    acc <= 48'sd0;
                    if (row + 1'b1 == n_rows) begin
                        busy <= 1'b0; done <= 1'b1;
                    end else begin
                        row   <= row + 1'b1;
                        p     <= row_ptr[row + 1'b1];
                        p_end <= row_ptr[row + 2];
                    end
                end else begin
                    acc <= acc + lane_sum;
                    p   <= p + P;
                end
            end
        end
    end
endmodule
