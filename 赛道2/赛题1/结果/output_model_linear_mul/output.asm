// ============================================================
// CIM Compiler Generated Assembly
// Track 2 Problem 1 - CNN Compiler
// ============================================================

//  // Process bit 0 of input (i8)
cim.bit.i8 /*out*/320, /*input*/0, /*index*/0, /*weightPos*/[0, 0, 64, 512]
//  // Process bit 1 of input, shift by 1
cim.bit.i8 /*out*/64, /*input*/0, /*index*/1, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/2, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 2 of input, shift by 2
cim.bit.i8 /*out*/64, /*input*/0, /*index*/2, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/4, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 3 of input, shift by 3
cim.bit.i8 /*out*/64, /*input*/0, /*index*/3, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/8, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 4 of input, shift by 4
cim.bit.i8 /*out*/64, /*input*/0, /*index*/4, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/16, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 5 of input, shift by 5
cim.bit.i8 /*out*/64, /*input*/0, /*index*/5, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/32, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 6 of input, shift by 6
cim.bit.i8 /*out*/64, /*input*/0, /*index*/6, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/64, /*len*/64
elt.add.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Process bit 7 (sign bit) of input, shift by 7 and subtract
cim.bit.i8 /*out*/64, /*input*/0, /*index*/7, /*weightPos*/[0, 0, 64, 512]
elt.mul.i32.vi /*dst*/576, /*src1*/64, /*src2*/128, /*len*/64
elt.sub.i32.vv /*dst*/320, /*src1*/320, /*src2*/576, /*len*/64
//  // Copy accumulated result to matmul_out
mem.copy.i32.i32 /*dst*/64, /*src*/320, /*len*/64
//  // Elementwise mul immediate 2
elt.mul.i32.vi /*dst*/832, /*src1*/64, /*src2*/2, /*len*/64

// End of generated code
