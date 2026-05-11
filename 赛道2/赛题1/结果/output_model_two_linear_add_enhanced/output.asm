// ============================================================
// CIM Compiler Generated Assembly
// Track 2 Problem 1 - CNN Compiler
// ============================================================

//  // Process bit 0 of input (i8)
cim.bit.i8 /*out*/388, /*input*/256, /*index*/0, /*weightPos*/[0, 0, 128, 512]
//  // Process bit 1 of input, shift by 1
cim.bit.i8 /*out*/0, /*input*/256, /*index*/1, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/2, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 2 of input, shift by 2
cim.bit.i8 /*out*/0, /*input*/256, /*index*/2, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/4, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 3 of input, shift by 3
cim.bit.i8 /*out*/0, /*input*/256, /*index*/3, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/8, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 4 of input, shift by 4
cim.bit.i8 /*out*/0, /*input*/256, /*index*/4, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/16, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 5 of input, shift by 5
cim.bit.i8 /*out*/0, /*input*/256, /*index*/5, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/32, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 6 of input, shift by 6
cim.bit.i8 /*out*/0, /*input*/256, /*index*/6, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/64, /*len*/128
elt.add.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Process bit 7 (sign bit) of input, shift by 7 and subtract
cim.bit.i8 /*out*/0, /*input*/256, /*index*/7, /*weightPos*/[0, 0, 128, 512]
elt.mul.i32.vi /*dst*/644, /*src1*/0, /*src2*/128, /*len*/128
elt.sub.i32.vv /*dst*/388, /*src1*/388, /*src2*/644, /*len*/128
//  // Copy accumulated result to hidden
mem.copy.i32.i32 /*dst*/0, /*src*/388, /*len*/128
//  // Elementwise add hidden add_const
elt.add.i32.vv /*dst*/256, /*src1*/0, /*src2*/384, /*len*/32

// End of generated code
