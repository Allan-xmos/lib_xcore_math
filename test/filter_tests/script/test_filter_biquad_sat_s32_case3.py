# Copyright 2020-2022 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
import numpy as np

Q30 = lambda x: np.int64(np.round(x * (2 ** 30)))
F30 = lambda x: x / 2 ** 30


def mul_q30(x, y):
    p = ((x * y) + (1 << 29)) >> 30
    # note this is 2**33
    assert (-0x200000000 <= p) and (p < 0x200000000)
    return p


def macc_q30(acc, x, y):
    p = mul_q30(x, y)
    acc += p
    assert (-2**39 <= acc) and (acc < 2**39-1)
    return acc


def acc_sat(acc):
    if acc > (2**31) - 1:
        acc = 2**31 - 1
        print("saturated")
    elif acc < -(2**31):
        acc = -(2**31)
        print("saturated")
    return acc


def biquad_q30(smp_in, state, coef):
    acc = 0
    for k in [4, 3, 2, 1]:
        acc = macc_q30(acc, coef[k], state[k - 1])
    acc = macc_q30(acc, coef[0], smp_in)
    acc = acc_sat(acc)

    state[1] = state[0]
    state[0] = smp_in
    state[3] = state[2]
    state[2] = acc
    return acc


def biquad(smp_in, state, coef):
    tmp = coef[0] * smp_in
    tmp += coef[1] * state[0] + coef[2] * state[1]
    tmp += coef[3] * state[2] + coef[4] * state[3]
    state[1] = state[0]
    state[0] = smp_in
    state[3] = state[2]
    state[2] = tmp
    return tmp


def biquads(smp_in, N, states, coefs):
    for k in range(N):
        smp_in = biquad(smp_in, states[k], coefs[k])
    return smp_in


def biquads_q30(smp_in, N, states, coefs):
    for k in range(N):
        smp_in = biquad_q30(smp_in, states[k], coefs[k])
    return smp_in


N_biquads = 8

coef = np.ones(shape=(N_biquads, 5))

# biquads in the order
# [['notch', fs*0.05, 1],
# ['notch', fs*0.10, 1],
# ['notch', fs*0.15, 1],
# ['notch', fs*0.20, 1],
# ['notch', fs*0.25, 1],
# ['notch', fs*0.30, 1],
# ['notch', fs*0.35, 1],
# ['lowshelf', fs*1000/48000, 1, 3]]
coef[0, :] = [0.8661694582551718, -1.6475522154942155, 0.8661694582551718, 1.6475522154942155, -0.7323389174416661]
coef[1, :] = [0.7728616576641798, -1.2505164304748178, 0.7728616576641798, 1.2505164304748178, -0.5457233153283596]
coef[2, :] = [0.7119928449392319, -0.8369977874681354, 0.7119928449392319, 0.8369977874681354, -0.42398568894714117]
coef[3, :] = [0.6777233807370067, -0.41885608434677124, 0.6777233807370067, 0.41885608434677124, -0.3554467624053359]
coef[4, :] = [0.6666666669771075, 0.0, 0.6666666669771075, 0.0, -0.3333333330228925]
coef[5, :] = [0.6777233807370067, 0.41885608434677124, 0.6777233807370067, -0.41885608434677124, -0.3554467624053359]
coef[6, :] = [0.7119928449392319, 0.8369977874681354, 0.7119928449392319, -0.8369977874681354, -0.42398568894714117]
coef[7, :] = [1.012055790051818, -1.8705660654231906, 0.8777095526456833, 1.8733696807175875, -0.8869617283344269]

coef_q30 = Q30(coef)

state = np.zeros(shape=(N_biquads, 4))
state_q30 = np.zeros(shape=(N_biquads, 4), dtype=np.int64)


# N_samples = np.random.randint(0, 200)

# alternate through ±((2**n) - 1)
samples = 2**(np.arange(0, 31.5, 0.5)) - 1
samples = np.round(np.repeat(samples, 2)).astype(int)
samples[::2] *= -1

N_samples = len(samples)

y = np.zeros(shape=samples.shape)
y_q30 = np.zeros(shape=samples.shape, dtype=np.int64)


# print(f"N_samples: {N_samples}")

for i in range(len(samples)):
    y[i] = biquads(samples[i], N_biquads, state, coef)
    y_q30[i] = biquads_q30(samples[i], N_biquads, state_q30, coef_q30)

diff = y - y_q30


# print("y[] = ", y)
# print("y_q30[] = ", y_q30)
# print("\n\n")

asd = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

print(f"Max absolute difference: y[{asd}] - y_q30[{asd}] = {y[asd]-y_q30[asd]}")

# print(f"Badness:  {np.max(np.abs(diff)) / np.mean(np.abs(y))}"     )

# print(diff)

print("\n\n")


pcoef = np.zeros(shape=(5, 8), dtype=np.int64)
pcoef[:, :N_biquads] = coef_q30.transpose()


def array_to_string(arr, val_fmtr=str):
    if len(arr.shape) == 1:
        return f"{{{', '.join([val_fmtr(x) for x in arr])}}}"
    else:
        return (
            "{"
            + ",".join([array_to_string(arr[i], val_fmtr) for i in range(arr.shape[0])])
            + "}"
        )


def print_arr_c(name, arr, val_fmtr=str):
    dims = "".join([f"[{val_fmtr(x)}]" for x in arr.shape])
    print(f"int32_t {name}{dims} = {array_to_string(arr,val_fmtr=val_fmtr)};")


def neg_hex(val):
    return (hex(val & (2**32-1)))



print("    ", end="")
print_arr_c("samples", samples.astype(np.int64))
print("    ", end="")
print_arr_c("coef", pcoef, val_fmtr=neg_hex)
print("    ", end="")
print_arr_c("Y_exp", y_q30.astype(np.int64))


# print( "    filter_biquad_s32_t filter;")
# print(f"    filter.biquad_count = {N_biquads};")
# print( "    memset(&filter.state, 0, sizeof(filter.state));")
# print( "    memcpy(&filter.coef, coef, sizeof(coef));")

