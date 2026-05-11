// ============================================================
// CIM Compiler Generated Assembly
// Track 2 Problem 1 - CNN Compiler
// ============================================================

//  // Process bit 0 of input (i8)
cim.bit.i8 /*out*/384, /*input*/0, /*index*/0, /*weightPos*/[0, 0, 128, 512]
//  // Process bit 1 of input, shift by 1
cim.bit.i8 /*out*/128, /*input*/0, /*index*/1, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/2, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 2 of input, shift by 2
cim.bit.i8 /*out*/128, /*input*/0, /*index*/2, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/4, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 3 of input, shift by 3
cim.bit.i8 /*out*/128, /*input*/0, /*index*/3, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/8, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 4 of input, shift by 4
cim.bit.i8 /*out*/128, /*input*/0, /*index*/4, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/16, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 5 of input, shift by 5
cim.bit.i8 /*out*/128, /*input*/0, /*index*/5, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/32, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 6 of input, shift by 6
cim.bit.i8 /*out*/128, /*input*/0, /*index*/6, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/64, /*len*/128
elt.add.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Process bit 7 (sign bit) of input, shift by 7 and subtract
cim.bit.i8 /*out*/128, /*input*/0, /*index*/7, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/640, /*src1*/128, /*src2*/128, /*len*/128
elt.sub.i32.vv /*dst*/384, /*src1*/384, /*src2*/640, /*len*/128
//  // Copy accumulated result to hidden
mem.copy.i32.i32 /*dst*/128, /*src*/384, /*len*/128
//  // Process bit 0 of hidden (i32)
cim.bit.i32 /*out*/1024, /*input*/128, /*index*/0, /*weightPos*/[0, 0, 64, 256]
//  // Process bit 1 of hidden, shift by 1
cim.bit.i32 /*out*/896, /*input*/128, /*index*/1, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/2, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 2 of hidden, shift by 2
cim.bit.i32 /*out*/896, /*input*/128, /*index*/2, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/4, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 3 of hidden, shift by 3
cim.bit.i32 /*out*/896, /*input*/128, /*index*/3, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/8, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 4 of hidden, shift by 4
cim.bit.i32 /*out*/896, /*input*/128, /*index*/4, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/16, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 5 of hidden, shift by 5
cim.bit.i32 /*out*/896, /*input*/128, /*index*/5, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/32, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 6 of hidden, shift by 6
cim.bit.i32 /*out*/896, /*input*/128, /*index*/6, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/64, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 7 of hidden, shift by 7
cim.bit.i32 /*out*/896, /*input*/128, /*index*/7, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/128, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 8 of hidden, shift by 8
cim.bit.i32 /*out*/896, /*input*/128, /*index*/8, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/256, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 9 of hidden, shift by 9
cim.bit.i32 /*out*/896, /*input*/128, /*index*/9, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/512, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 10 of hidden, shift by 10
cim.bit.i32 /*out*/896, /*input*/128, /*index*/10, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/1024, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 11 of hidden, shift by 11
cim.bit.i32 /*out*/896, /*input*/128, /*index*/11, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/2048, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 12 of hidden, shift by 12
cim.bit.i32 /*out*/896, /*input*/128, /*index*/12, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/4096, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 13 of hidden, shift by 13
cim.bit.i32 /*out*/896, /*input*/128, /*index*/13, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/8192, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 14 of hidden, shift by 14
cim.bit.i32 /*out*/896, /*input*/128, /*index*/14, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/16384, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 15 of hidden, shift by 15
cim.bit.i32 /*out*/896, /*input*/128, /*index*/15, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/32768, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 16 of hidden, shift by 16
cim.bit.i32 /*out*/896, /*input*/128, /*index*/16, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/65536, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 17 of hidden, shift by 17
cim.bit.i32 /*out*/896, /*input*/128, /*index*/17, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/131072, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 18 of hidden, shift by 18
cim.bit.i32 /*out*/896, /*input*/128, /*index*/18, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/262144, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 19 of hidden, shift by 19
cim.bit.i32 /*out*/896, /*input*/128, /*index*/19, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/524288, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 20 of hidden, shift by 20
cim.bit.i32 /*out*/896, /*input*/128, /*index*/20, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/1048576, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 21 of hidden, shift by 21
cim.bit.i32 /*out*/896, /*input*/128, /*index*/21, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/2097152, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 22 of hidden, shift by 22
cim.bit.i32 /*out*/896, /*input*/128, /*index*/22, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/4194304, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 23 of hidden, shift by 23
cim.bit.i32 /*out*/896, /*input*/128, /*index*/23, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/8388608, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 24 of hidden, shift by 24
cim.bit.i32 /*out*/896, /*input*/128, /*index*/24, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/16777216, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 25 of hidden, shift by 25
cim.bit.i32 /*out*/896, /*input*/128, /*index*/25, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/33554432, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 26 of hidden, shift by 26
cim.bit.i32 /*out*/896, /*input*/128, /*index*/26, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/67108864, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 27 of hidden, shift by 27
cim.bit.i32 /*out*/896, /*input*/128, /*index*/27, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/134217728, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 28 of hidden, shift by 28
cim.bit.i32 /*out*/896, /*input*/128, /*index*/28, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/268435456, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 29 of hidden, shift by 29
cim.bit.i32 /*out*/896, /*input*/128, /*index*/29, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/536870912, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 30 of hidden, shift by 30
cim.bit.i32 /*out*/896, /*input*/128, /*index*/30, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/1073741824, /*len*/64
elt.add.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Process bit 31 (sign bit) of hidden, shift by 31 and subtract
cim.bit.i32 /*out*/896, /*input*/128, /*index*/31, /*weightPos*/[0, 0, 64, 256]
elt.mul.i32.vi /*dst*/1152, /*src1*/896, /*src2*/2147483648, /*len*/64
elt.sub.i32.vv /*dst*/1024, /*src1*/1024, /*src2*/1152, /*len*/64
//  // Copy accumulated result to matmul_out
mem.copy.i32.i32 /*dst*/896, /*src*/1024, /*len*/64
//  // Elementwise add immediate 1
elt.add.i32.vi /*dst*/1280, /*src1*/896, /*src2*/1, /*len*/32

// End of generated code
