// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__VX4B__)


/*
int64_t vect_s16_dot(
    const int16_t b[],
    const int16_t c[],
    const unsigned length);
*/


/*
    Like the XS3 version this keeps a 48-bit accumulator for each of the 16 VPU
    lanes, but VX4 has no VADDDR, so the lanes cannot be collapsed with a single
    vector instruction. The accumulators live in three 16-bit planes in memory,

        acc[k] = ACC_LOW[k] + ((ACC_MID[k] + (ACC_HIGH[k] << 16)) << 15)

    with ACC_LOW[k] and ACC_MID[k] unsigned and ACC_HIGH[k] signed, and the 16
    lanes are summed scalar-side at the end.

    Each iteration multiply-accumulates one vector of products into ACC_LOW,
    starting from a zeroed vD so that vD is left holding just the carry out of
    the low plane, and then folds that carry into the 32-bit ACC_MID:ACC_HIGH
    accumulator pair.

    The carry is folded in with a vector of 2s rather than 1s, which is why the
    upper planes are worth 2^15 instead of 2^16 above. A VX4 16-bit multiply
    accumulate rounds its 32-bit result down to even, so an odd sum loses a
    count; the products of ACC_LOW (worth 1) can absorb that, but the carries
    (worth 2^16) cannot. Scaling the carries by two keeps every addend -- and
    therefore the running accumulator -- even, so the fold is exact. The cost is
    one bit of range in the carry plane: a lane can take at most 65535 products
    of 2^30 before ACC_MID:ACC_HIGH saturates (at the odd value 0x7FFFFFFF), so
    `length` is limited to 16*65535 = 1048560 rather than the ~2 million of the
    XS3 implementation.

    Because ACC_LOW is always even, the residual error is exactly one count for
    each k where b[k] and c[k] are both odd, so the result is deterministic and
    at most `length` below the true inner product.
*/


#include "../asm_helper.h"

.text
.p2align 2

#define NSTACKVECS      (5)
// the 4 spare words at the top of the frame keep the vectors clear of the link
// register that xm.entsp pushes there
#define NSTACKWORDS     (8 + 8*NSTACKVECS + 4)

#define FUNCTION_NAME   vect_s16_dot

// The three accumulator planes must stay contiguous and in this order -- the
// final reduction reaches ACC_MID and ACC_HIGH by immediate offset from ACC_LOW.
#define STACK_ACC_LOW       (NSTACKWORDS-8*5-4)
#define STACK_ACC_MID       (NSTACKWORDS-8*4-4)
#define STACK_ACC_HIGH      (NSTACKWORDS-8*3-4)
#define STACK_VEC_CARRY     (NSTACKWORDS-8*2-4)
#define STACK_VEC_TAIL      (NSTACKWORDS-8*1-4)

#define OFFSET_ACC_MID      (32)
#define OFFSET_ACC_HIGH     (64)

#define b           a0
#define c           a1
#define N           a2
#define tail        a3
#define acc_low     s2
#define acc_mid     s3
#define acc_high    s4
#define carry       s5
#define vec_twos    s6
#define vec_tail    s7
#define _32         t3

// scalar accumulators used by the final reduction
#define sum_low     a0
#define sum_mid     a1
#define sum_high    a2

.globl FUNCTION_NAME
.type FUNCTION_NAME,@function

FUNCTION_NAME:
        xm.entsp (NSTACKWORDS)*4
        xm.stdsp  s3,s2,0
        xm.stdsp  s5,s4,8
        xm.stdsp  s7,s6,16
        li t3, 0x0100

    { slli tail, N, SIZEOF_LOG2_S16     ; xm.vsetc t3                           }
    { xm.zexti tail, 5                  ; xm.vclrdr                             }
    { srli N, N, EPV_LOG2_S16           ; nop                                   }
        addi acc_low,  sp, (STACK_ACC_LOW)*4
        addi acc_mid,  sp, (STACK_ACC_MID)*4
        addi acc_high, sp, (STACK_ACC_HIGH)*4
        addi carry,    sp, (STACK_VEC_CARRY)*4
        addi vec_tail, sp, (STACK_VEC_TAIL)*4
        la vec_twos, vpu_vec_0x0002

    // vR is zero, so these three stores clear the 48-bit accumulators
    { xm.mkmsk tail, tail               ; xm.vstr acc_low                       }
    { li _32, 32                        ; xm.vstr acc_mid                       }
    { nop                               ; xm.vstr acc_high                      }
    { nop                               ; xm.brff N, .L_loop_bot                }

.L_loop_top:
        { nop                           ; xm.vclrdr                             }
        { nop                           ; xm.vldr acc_low                       }
        { add b, b, _32                 ; xm.vldc b                             }
        { nop                           ; xm.vlmacc0 c                          }
        xm.vlmacc1 c
        { add c, c, _32                 ; xm.vstr acc_low                       }
        { nop                           ; xm.vstd carry                         }

        // fold the carry out of the low plane into ACC_MID:ACC_HIGH
        { nop                           ; xm.vldr acc_mid                       }
        { nop                           ; xm.vldd acc_high                      }
        { nop                           ; xm.vldc vec_twos                      }
        { nop                           ; xm.vlmacc0 carry                      }
        xm.vlmacc1 carry
        { addi N, N, -1                 ; xm.vstr acc_mid                       }
        { nop                           ; xm.vstd acc_high                      }
        { nop                           ; xm.bt N, .L_loop_top                  }
.L_loop_bot:

    // The final partial vector is zero-padded into a scratch vector and then run
    // through the loop once more. c is read unmasked; those lanes multiply zeros.
    { nop                               ; xm.brff tail, .L_reduce               }
    { nop                               ; xm.vclrdr                             }
    { nop                               ; xm.vstd vec_tail                      }
    { nop                               ; xm.vldr b                             }
        xm.vstrpv vec_tail, tail
    { mv b, vec_tail                    ; li N, 1                               }
    { li tail, 0                        ; xm.bu .L_loop_top                     }

.L_reduce:
    // Sum each plane across the 16 lanes. The two lower planes hold unsigned
    // values, the top plane signed ones.
    { li sum_low, 0                     ; nop                                   }
    { li sum_mid, 0                     ; nop                                   }
    { li sum_high, 0                    ; nop                                   }

        .set VS16DOT_K, 0
        .rept 16
        lhu t3, VS16DOT_K(acc_low)
        add sum_low, sum_low, t3
        lhu t3, VS16DOT_K+OFFSET_ACC_MID(acc_low)
        add sum_mid, sum_mid, t3
        lh  t3, VS16DOT_K+OFFSET_ACC_HIGH(acc_low)
        add sum_high, sum_high, t3
        .set VS16DOT_K, VS16DOT_K+2
        .endr

.L_finish:
        xm.lddsp  s3,s2,0
        xm.lddsp  s5,s4,8
        xm.lddsp  s7,s6,16

    // a = sum_low + (sum_mid << 15) + (sum_high << 31)
    //
    // sum_high sums 16 lanes, so it can outgrow 16 bits and neither shift can be
    // done in 32 bits. Instead each term's overflow is moved up into the next,
    // which leaves three disjoint bit fields in the low word:
    //
    //   sum_mid'  = sum_mid  + (sum_low  >> 15)
    //   sum_high' = sum_high + (sum_mid' >> 16)
    //   a         = (sum_low & 0x7FFF) | ((sum_mid' & 0xFFFF) << 15) | (sum_high' << 31)
        srli t3, sum_low, 15
    { add sum_mid, sum_mid, t3          ; nop                                   }
        slli sum_low, sum_low, 17           // xm.zexti can't take a width of 15
        srli sum_low, sum_low, 17
    { srli t3, sum_mid, 16              ; nop                                   }
    { add sum_high, sum_high, t3        ; nop                                   }
        xm.zexti sum_mid, 16
        slli sum_mid, sum_mid, 15
    { add sum_low, sum_low, sum_mid     ; nop                                   }
        slli t3, sum_high, 31
        srai a1, sum_high, 1
    { add a0, sum_low, t3               ; xm.retsp (NSTACKWORDS)*4              }

.resource_const FUNCTION_NAME, "stack_frame_bytes", (NSTACKWORDS)*4
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"
.L_end:
    .size FUNCTION_NAME, .L_end - FUNCTION_NAME


#endif //defined(__VX4B__)
