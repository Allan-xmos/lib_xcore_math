// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#include "xmath/xmath.h"

#include "../tst_common.h"

#include "unity_fixture.h"


TEST_GROUP_RUNNER(vect_dot) {
  RUN_TEST_CASE(vect_dot, vect_s32_dot_prepare);
  RUN_TEST_CASE(vect_dot, vect_s16_dot);
  RUN_TEST_CASE(vect_dot, vect_s16_dot_basic);
  RUN_TEST_CASE(vect_dot, vect_s16_dot_long);
  RUN_TEST_CASE(vect_dot, vect_s32_dot_basic);
}

TEST_GROUP(vect_dot);
TEST_SETUP(vect_dot) { fflush(stdout); }
TEST_TEAR_DOWN(vect_dot) {}


static char msg_buff[200];


#define TEST_ASSERT_EQUAL_MSG(EXPECTED, ACTUAL, LINE_NUM)   do{       \
    if((EXPECTED)!=(ACTUAL)) {                                        \
      sprintf(msg_buff, "(test vector @ line %u)", (LINE_NUM));       \
      TEST_ASSERT_EQUAL_MESSAGE((EXPECTED), (ACTUAL), msg_buff);      \
    }} while(0)


/**
 * A VX4 16-bit multiply-accumulate drops the LSB of its result, so every odd
 * product costs the inner product a count. `vect_s16_dot()` is therefore never
 * above the true result, and never further than `LENGTH` below it. The bound has
 * to scale with the length -- a fixed tolerance would be met only by chance.
 */
#if defined(__VX4B__)
#  define TEST_ASSERT_DOT_S16(EXPECTED, ACTUAL, LENGTH) do{                          \
      if( ((ACTUAL) > (EXPECTED)) || (((EXPECTED)-(ACTUAL)) > (int64_t)(LENGTH)) ){  \
        sprintf(msg_buff, "(length %u; expected %lld; got %lld)",                    \
                (unsigned) (LENGTH), (long long) (EXPECTED), (long long) (ACTUAL));  \
        TEST_FAIL_MESSAGE(msg_buff);                                                 \
      }} while(0)
#else
#  define TEST_ASSERT_DOT_S16(EXPECTED, ACTUAL, LENGTH) do{                          \
      if((EXPECTED) != (ACTUAL)){                                                    \
        sprintf(msg_buff, "(length %u; expected %lld; got %lld)",                    \
                (unsigned) (LENGTH), (long long) (EXPECTED), (long long) (ACTUAL));  \
        TEST_FAIL_MESSAGE(msg_buff);                                                 \
      }} while(0)
#endif



#if SMOKE_TEST
#  define REPS       (100)
#  define MAX_LEN    (64)
#else
#  define REPS       (1000)
#  define MAX_LEN    (256)
#endif


TEST(vect_dot, vect_s32_dot_prepare)
{
    
    unsigned seed = SEED_FROM_FUNC_NAME();


    int32_t B[MAX_LEN], C[MAX_LEN];

    for(int r = 0; r < REPS; r++){
        setExtraInfo_RS(r, seed);

        const unsigned B_length = pseudo_rand_uint(&seed, 1, MAX_LEN);

        const exponent_t B_exp = pseudo_rand_int(&seed, -100, 100);
        const exponent_t C_exp = pseudo_rand_int(&seed, -100, 100);

        const headroom_t B_hr = pseudo_rand_uint(&seed, 0, 30);
        const headroom_t C_hr = pseudo_rand_uint(&seed, 0, 30);

        for(unsigned int i = 0; i < B_length; i++){
            B[i] = INT32_MIN >> B_hr;
            C[i] = INT32_MIN >> C_hr;
        }

        double expected = B_length * (ldexp(B[0], B_exp) * ldexp(C[0], C_exp));

        exponent_t A_exp;
        right_shift_t b_shr, c_shr;

        vect_s32_dot_prepare(&A_exp, &b_shr, &c_shr, B_exp, C_exp, B_hr, C_hr, B_length);

        int64_t result = vect_s32_dot(B, C, B_length, b_shr, c_shr);

        double got = ldexp((double) result, A_exp);

        TEST_ASSERT( fabs((expected-got)/expected) < ldexp(1, -25) );
    }
}
#undef MAX_LEN
#undef REPS



#if SMOKE_TEST
#  define REPS       (100)
#  define MAX_LEN    (64)
#else
#  define REPS       (1000)
#  define MAX_LEN    (256)
#endif

TEST(vect_dot, vect_s16_dot)
{
    
    unsigned seed = SEED_FROM_FUNC_NAME();


    int16_t WORD_ALIGNED B[MAX_LEN];
    int16_t WORD_ALIGNED C[MAX_LEN];
    

    for(unsigned int v = 0; v < REPS; v++){

        setExtraInfo_RS(v, seed);

        const unsigned len = pseudo_rand_uint(&seed, 1, MAX_LEN+1);


        headroom_t B_hr = pseudo_rand_uint(&seed, 0, 15);
        headroom_t C_hr = pseudo_rand_uint(&seed, 0, 15);

        int64_t expected = 0;
        
        for(unsigned int i = 0; i < len; i++){
            B[i] = pseudo_rand_int16(&seed) >> B_hr;
            C[i] = pseudo_rand_int16(&seed) >> C_hr;

            expected += ((int32_t)B[i]) * C[i];
        }

        int64_t result = vect_s16_dot(B, C, len);

        TEST_ASSERT_DOT_S16(expected, result, len);
    }
}
#undef MAX_LEN
#undef REPS



/**
 * Fill the stack below the caller with non-zero junk, so that a function called
 * straight afterwards cannot get away with reading scratch space it never
 * initialised.
 */
static void __attribute__((noinline)) poison_stack(void)
{
    volatile int16_t junk[512];
    for(unsigned int i = 0; i < 512; i++)
        junk[i] = INT16_MAX;
}


#define MAX_LEN     (64)

/**
 * Deterministic corner cases: every length around the 16-element vector boundary
 * crossed with the extremes of the input range.
 *
 * The elements past `length` are poisoned with the largest-magnitude product
 * available, so that a vector implementation which fails to mask off the final
 * partial vector gives an obviously wrong answer rather than a plausible one. The
 * stack is poisoned too, for any scratch vectors the implementation keeps there.
 */
TEST(vect_dot, vect_s16_dot_basic)
{
    const unsigned lengths[] = { 0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 63, MAX_LEN };

    const struct {
        int16_t b;
        int16_t c;
    } values[] = {
        {         0,         0 },
        {         1,         1 },
        {        -1,        -1 },
        {         1,        -1 },
        { INT16_MAX, INT16_MAX },   // largest positive product; every product odd
        { INT16_MIN, INT16_MIN },   // largest product of all; every product even
        { INT16_MIN, INT16_MAX },   // largest negative product
        { INT16_MAX, INT16_MIN },
        {        -1, INT16_MIN },
    };

    int16_t WORD_ALIGNED B[MAX_LEN];
    int16_t WORD_ALIGNED C[MAX_LEN];

    const unsigned N_lengths = sizeof(lengths)/sizeof(lengths[0]);
    const unsigned N_values  = sizeof(values)/sizeof(values[0]);

    for(unsigned int l = 0; l < N_lengths; l++){
        const unsigned len = lengths[l];

        for(unsigned int v = 0; v < N_values; v++){
            setExtraInfo_R(l * N_values + v);

            for(unsigned int i = 0; i < MAX_LEN; i++){
                B[i] = INT16_MAX;
                C[i] = INT16_MIN;
            }

            for(unsigned int i = 0; i < len; i++){
                B[i] = values[v].b;
                C[i] = values[v].c;
            }

            const int64_t expected = ((int64_t) len) * values[v].b * values[v].c;

            poison_stack();
            const int64_t result = vect_s16_dot(B, C, len);

            TEST_ASSERT_DOT_S16(expected, result, len);
        }
    }
}
#undef MAX_LEN



// Long enough for the inner product to exceed 2^46 in magnitude, and not a multiple
// of 16 so that the final partial vector is exercised too.
#define LONG_LEN    (65536 + 37)

// b[] and c[] share this buffer, offset by two elements. The spare elements on the
// end cover that offset and a full vector of over-read past `length`.
static int16_t WORD_ALIGNED long_buff[LONG_LEN + 2 + 16];

/**
 * Inner products at the top of the documented range, where the 16 lane
 * accumulators sum to more than 32 bits even after dropping their low bits.
 */
TEST(vect_dot, vect_s16_dot_long)
{
    // INT16_MIN, INT16_MIN, INT16_MAX, INT16_MAX, ...
    for(unsigned int i = 0; i < sizeof(long_buff)/sizeof(long_buff[0]); i++)
        long_buff[i] = (i & 2)? INT16_MAX : INT16_MIN;

    // An offset of 0 squares each element; an offset of 2 pairs every INT16_MIN
    // with an INT16_MAX, for the largest negative result.
    for(unsigned int offset = 0; offset <= 2; offset += 2){
        setExtraInfo_R(offset);

        const int16_t* b = &long_buff[0];
        const int16_t* c = &long_buff[offset];

        int64_t expected = 0;
        for(unsigned int i = 0; i < LONG_LEN; i++)
            expected += ((int32_t) b[i]) * c[i];

        const int64_t result = vect_s16_dot(b, c, LONG_LEN);

        TEST_ASSERT_DOT_S16(expected, result, LONG_LEN);
    }
}
#undef LONG_LEN


#define MAX_LEN     40

TEST(vect_dot, vect_s32_dot_basic)
{
    

    typedef struct {
        struct{ int32_t b;  int32_t c;  } input;
        struct{ int b;      int c;      } shr;
        unsigned len;

        int64_t expected;

        unsigned line;
    } test_case_t;

    test_case_t casses[] = {
        //  input{           b,           c },  shr{   b,    c },   len,        expected,       line
        {        {  0x00000000,  0x00000000 },     {   0,    0 },     1,      0x00000000,     __LINE__ },
        {        {  0x00000000,  0x00000000 },     {   0,    0 },     2,      0x00000000,     __LINE__ },
        {        {  0x00000000,  0x00000000 },     {   0,    0 },    16,      0x00000000,     __LINE__ },
        {        {  0x00000000,  0x00000000 },     {   0,    0 },    32,      0x00000000,     __LINE__ },
        {        {  0x00000000,  0x00000000 },     {   0,    0 },    40,      0x00000000,     __LINE__ },
        {        {  0x00010000,  0x00000000 },     {   0,    0 },     1,      0x00000000,     __LINE__ },
        {        {  0x00000000,  0x00100000 },     {   0,    0 },     2,      0x00000000,     __LINE__ },
        {        {  0x22000000,  0x00000000 },     {   0,    0 },    16,      0x00000000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },     1,      0x00010000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },     2,      0x00020000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },     8,      0x00080000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },    16,      0x00100000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },    32,      0x00200000,     __LINE__ },
        {        {  0x40000000,  0x00010000 },     {   0,    0 },    40,      0x00280000,     __LINE__ },
        {        {  0x04000000,  0x01000000 },     {   0,    0 },     1,      0x00100000,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {   0,    0 },    25,      0x00190000,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {   0,    0 },    25,      0x00190000,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {   4,    4 },    25,      0x00001900,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {   2,    2 },    25,      0x00019000,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {  -1,    0 },    25,      0x00320000,     __LINE__ },
        {        {  0x04000000,  0x00100000 },     {  -1,   -1 },    25,      0x00640000,     __LINE__ },
    };

    const unsigned start_case = 0;

    const unsigned N_cases = sizeof(casses)/sizeof(test_case_t);
    for(unsigned int v = start_case; v < N_cases; v++){
        setExtraInfo_R(v);
        
        test_case_t* casse = &casses[v];

        unsigned len = casse->len;

        TEST_ASSERT(len <= MAX_LEN);

        int32_t WORD_ALIGNED B[MAX_LEN];
        int32_t WORD_ALIGNED C[MAX_LEN];
        int64_t result;

        for(unsigned int i = 0; i < len; i++){
            B[i] = casse->input.b;
            C[i] = casse->input.c;
        }

        result = vect_s32_dot(B, C, len, casse->shr.b, casse->shr.c);

        TEST_ASSERT_EQUAL_MSG(casse->expected, result, casse->line);
    }
}
#undef MAX_LEN

