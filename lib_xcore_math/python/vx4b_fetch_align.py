# Copyright 2026 XMOS LIMITED.
# This Software is subject to the terms of the XMOS Public Licence: Version 1.
"""Check (or fix) the 32-byte fetch phase of the VX4 assembly functions in lib_xcore_math.

VX4 fetches instructions in 32-byte aligned words. A taken branch fetches the word that holds its target, and the
next word is only fetched in an issue slot whose bundle does not use the memory port. If a loop starts with several
memory bundles and its first free slot is not in the same word as the loop head, every iteration loses a slot to a
fetch-NOP. The speed of such a loop therefore depends on where its function is placed.

To make the timing independent of placement, every vx4b function whose loops are sensitive to their phase starts
with `.p2align 5`, followed by an optional `.skip P` (commented `// Tuned for speed`) that sets the phase P of the
function within a 32-byte word. P is chosen so that the loops lose as few slots as possible, inner loops first.

This script checks that P is still the best phase for every function. It links the library with every function at
each of the 16 even phases, scores the loops of each function with the fetch rule, and reports functions that
  - are sensitive to their phase but have no `.p2align 5`,
  - use a phase that is slower than another phase, or
  - use a phase at which they fail to link (a short branch out of range because of internal alignment).
With --fix the .S files are rewritten with the best phases.

Requires the XTC tools (xcc, xobjdump) for VX4 on PATH.

usage: python vx4b_fetch_align.py [--fix] [--target XK-EVK-XU416] [--jobs N] [--build-dir DIR] [-v]
"""
import argparse
import bisect
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
VX4B = Path('src', 'arch', 'vx4b')
C_DIRS = ['src/vect', 'src/bfp', 'src/dct', 'src/fft', 'src/filter', 'src/scalar', 'src/arch/vx4b', 'src/etc']
INCLUDES = ['api', 'src/vect', 'src/etc/xmath_fft_lut']
PHASES = range(0, 32, 2)
TUNED_COMMENT = '// Tuned for speed'

# --- fetch model ------------------------------------------------------------------------------------------------

MEM = re.compile(r'^(lw|sw|lb|lbu|lh|lhu|sb|sh|xm\.ld\w*|xm\.st\w*|xm\.entsp|xm\.retsp|xm\.vl\w*|xm\.vst\w*)$')
NOT_MEM = {'xm.ldap', 'xm.lda16b', 'xm.lda16f', 'xm.ldivu'}
BRANCH = re.compile(r'^(xm\.bt|xm\.bf|xm\.brff|xm\.brft|xm\.bu|xm\.bru|j|beq|beqz|bne|bnez|blt|bltu|bge|bgeu|bltz|'
                    r'bgez|blez|bgtz)$')
INSTR = re.compile(r'^\S*\s+0x([0-9a-f]+): ((?:[0-9a-f]{2} ?)+):\s*(.*)$')
SYMBOL = re.compile(r'^0x([0-9a-f]+)\s+g\s+f\s+\S+\s+0x([0-9a-f]+)\s+(\S+)$')


def parse_disassembly(text):
    """{address: instruction} for the tile 0 image of `xobjdump -d` output."""
    ins = {}
    on = False
    for line in text.splitlines():
        if line.startswith('Loadable'):
            on = line.startswith('Loadable 2 for tile[0]')
            continue
        m = INSTR.match(line) if on else None
        if not m:
            continue
        addr = int(m.group(1), 16)
        ops = [p.split() for p in m.group(3).strip().strip('{}').split(';') if p.split()]
        target = None
        for op in ops:
            t = re.search(r'(-?0x[0-9a-f]+) <', ' '.join(op)) if BRANCH.match(op[0]) else None
            if t:
                target = addr + int(t.group(1), 16)
        ins[addr] = dict(size=len(m.group(2).split()), target=target,
                         mem=any(MEM.match(op[0]) and op[0] not in NOT_MEM for op in ops))
    return ins


def parse_symbols(text):
    """{function: (address, size)} for the tile 0 image of `xobjdump -t` output."""
    syms = {}
    on = False
    for line in text.splitlines():
        if line.startswith('Loadable'):
            on = line.startswith('Loadable 2 for tile[0]')
            continue
        m = SYMBOL.match(line.strip()) if on else None
        if m:
            syms[m.group(3)] = (int(m.group(1), 16), int(m.group(2), 16))
    return syms


def fetch_nops(ins, head, back):
    """Fetch-NOPs in one iteration of the loop from `head` to the backward branch at `back`."""
    avail = (head // 32) * 32 + 32
    pc = head
    n = 0
    while pc <= back and pc in ins:
        end = pc + ins[pc]['size']
        while end > avail:  # the next bundle is not fully fetched: a fetch-NOP fetches the next word
            n += 1
            avail += 32
        if not ins[pc]['mem']:  # a free slot fetches the next word
            avail = max(avail, (pc // 32) * 32 + 64)
        pc = end
    return n


def branches(ins):
    """Sorted (pc, target) of every branch instruction."""
    return sorted((pc, i['target']) for pc, i in ins.items() if i['target'] is not None)


def score(ins, brs, start, size):
    """(weighted, total, loops) fetch-NOPs of the loops in a function; a loop nested in d others weighs 100**d."""
    lo = bisect.bisect_left(brs, (start, -1))
    hi = bisect.bisect_left(brs, (start + size, -1))
    loops = [(t, pc) for pc, t in brs[lo:hi] if start <= t <= pc]
    weighted = total = 0
    for head, back in loops:
        n = fetch_nops(ins, head, back)
        depth = sum(1 for h, b in loops if (h, b) != (head, back) and h <= head and back <= b)
        weighted += n * 100 ** depth
        total += n
    return weighted, total, len(loops)


# --- sources ----------------------------------------------------------------------------------------------------

DEFINE = re.compile(r'\s*#define\s+(\w+)\s+(\w+)\s*(//.*)?$')
LABEL = re.compile(r'^\s*(\w+):\s*(//.*)?$')
QUIET = re.compile(r'^\.(globl|global|type|text|section|issue_mode|cc_top|set|file|p2align|align|balign)\b')
P2ALIGN = re.compile(r'^\s*\.p2align\s+(\d+)\s*$')
SKIP = re.compile(r'^\s*\.skip\s+(\d+)\s*$')


def asm_sources(lib):
    return sorted(p.relative_to(lib) for p in (lib / VX4B).rglob('*.S'))


def functions(text):
    """Names of the functions defined in a .S file (resolving `#define MACRO name` used as the label)."""
    names, macro = set(), {}
    for line in text.split('\n'):
        m = DEFINE.match(line)
        if m:
            macro[m.group(1)] = m.group(2)
        m = re.match(r'\s*\.type\s+(\w+)\s*,\s*@function', line)
        if m:
            names.add(macro.get(m.group(1), m.group(1)))
        m = re.match(r'^\s*(\w+):', line)
        if m and m.group(1) in macro:
            names.add(macro[m.group(1)])
    return {n for n in names if n not in macro}


def _walk(text):
    """Yield (line, function label or None, emits_nothing) for each line of a .S file."""
    macro, in_block = {}, False
    for line in text.split('\n'):
        s = line.strip()
        quiet = in_block or not s or s.startswith(('//', '#', '/*')) or bool(QUIET.match(s))
        if '/*' in s and '*/' not in s.split('/*', 1)[1]:
            in_block = True
        if in_block and '*/' in s:
            in_block = False
        m = DEFINE.match(line)
        if m:
            macro[m.group(1)] = m.group(2)
        m = LABEL.match(line)
        yield line, (macro.get(m.group(1), m.group(1)) if m else None), quiet


def phases_in(text):
    """{function: committed phase P, or None if the function has no `.p2align 5` of its own}."""
    result, above = {}, []  # above: (line, quiet) since the last line that emits bytes, plus that line
    for line, label, quiet in _walk(text):
        if label:
            p = None
            skip = 0
            for prev, q in reversed(above):
                m = SKIP.match(prev)
                if m and not skip:
                    skip = int(m.group(1))
                    continue
                if not q:
                    break
                m = P2ALIGN.match(prev)
                if m:
                    p = skip if m.group(1) == '5' else None
                    break
            result[label] = p
        above = above + [(line, quiet)] if quiet or SKIP.match(line) else [(line, quiet)]
    return result


def rewrite(text, phase):
    """Set the phase of each function in `phase` ({function: P}): one `.p2align 5` above the label (upgrading the
    function's own alignment directive if nothing emits bytes in between) and, for P > 0, the tuned `.skip P`."""
    out, quiet_out = [], []
    lines = list(_walk(text))
    def tuned_skip_of_rewritten(k):  # line k is the `.skip` directly above a label in `phase`
        return bool(SKIP.match(lines[k][0])) and k + 1 < len(lines) and lines[k + 1][1] in phase

    for k, (line, label, quiet) in enumerate(lines):
        if tuned_skip_of_rewritten(k) or (line.strip() == TUNED_COMMENT and tuned_skip_of_rewritten(k + 1)):
            continue  # drop the old tuned skip and its comment (re-added below)
        if label in phase:
            i = len(out) - 1
            while i >= 0 and quiet_out[i] and not P2ALIGN.match(out[i]):
                i -= 1
            if i >= 0 and quiet_out[i]:
                out[i] = re.match(r'^(\s*)', out[i]).group(1) + '.p2align 5'
            else:
                out.append('.p2align 5')
                quiet_out.append(True)
            if phase[label]:
                out += [TUNED_COMMENT, f'.skip {phase[label]}']
                quiet_out += [True, False]
        out.append(line)
        quiet_out.append(quiet)
    return '\n'.join(out)


def read(path):
    raw = path.read_bytes().decode()
    return raw.replace('\r\n', '\n'), '\r\n' if '\r\n' in raw else '\n'


# --- build ------------------------------------------------------------------------------------------------------

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return r.returncode, r.stdout


def compile_all(jobs, items):
    """items: [(cmd, label)] -> first failure message or None."""
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for (rc, out), (cmd, label) in zip(pool.map(lambda it: run(it[0]), items), items):
            if rc:
                return f'{label} failed:\n{out}'
    return None


def link_trial(lib, work, target, jobs, c_objs, all_fns, phase_of):
    """Build the probe app with the given {function: P}; -> (ins, syms) or link-error text."""
    src = work / 'src'
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(lib / VX4B, src / VX4B, ignore=shutil.ignore_patterns('*.S'))
    objs = []
    items = []
    for rel in asm_sources(lib):
        text, _ = read(lib / rel)
        text = rewrite(text, {f: phase_of[f] for f in functions(text)})
        dst = src / rel
        dst.write_text(text)
        obj = work / 'obj' / (str(rel).replace(os.sep, '_') + '.o')
        objs.append(obj)
        items.append((['xcc', '-c', '-Os', f'-target={target}', *[f'-I{lib / i}' for i in INCLUDES],
                       '-x', 'assembler-with-cpp', str(dst), '-o', str(obj)], str(rel)))
    err = compile_all(jobs, items)
    if err:
        sys.exit(err)
    xe = work / 'probe.xe'
    rc, out = run(['xcc', f'-target={target}', *map(str, c_objs), *map(str, objs), '-o', str(xe)])
    if rc:
        return out
    return parse_disassembly(run(['xobjdump', '-d', str(xe)])[1]), parse_symbols(run(['xobjdump', '-t', str(xe)])[1])


def measure(lib, work, target, jobs, verbose):
    """{function: {P: (weighted, total, loops)}} and {function: [P that fail to link]}."""
    (work / 'obj').mkdir(parents=True, exist_ok=True)
    all_fns = {f: rel for rel in asm_sources(lib) for f in functions(read(lib / rel)[0])}
    probe = work / 'probe_main.c'
    probe.write_text(''.join(f'extern char {f}[] __attribute__((weak));\n' for f in sorted(all_fns)) +
                     'void *volatile probe_refs[] = {\n' + ''.join(f'    (void *){f},\n' for f in sorted(all_fns)) +
                     '};\nint main(void) {\n    unsigned s = 0;\n'
                     '    for (unsigned i = 0; i < sizeof(probe_refs) / sizeof(probe_refs[0]); i++)\n'
                     '        s += (unsigned)probe_refs[i];\n    return (int)s;\n}\n')
    c_srcs = [probe] + sorted(p for d in C_DIRS for p in (lib / d).rglob('*.c'))
    c_objs = [work / 'obj' / f'c{i}.o' for i in range(len(c_srcs))]
    err = compile_all(jobs, [(['xcc', '-c', '-Os', f'-target={target}', *[f'-I{lib / i}' for i in INCLUDES],
                               str(s), '-o', str(o)], str(s)) for s, o in zip(c_srcs, c_objs)])
    if err:
        sys.exit(err)
    scores = {f: {} for f in all_fns}
    unlinkable = {f: [] for f in all_fns}
    for P in PHASES:
        excluded = set()
        while True:
            # functions that do not link at P are placed at a phase that does, and not scored at P
            phase_of = {f: (next(q for q in PHASES if q not in unlinkable[f]) if f in excluded else P) for f in all_fns}
            res = link_trial(lib, work, target, jobs, c_objs, all_fns, phase_of)
            if not isinstance(res, str):
                break
            failing = set(re.findall(r'\(function (\w+):', res))
            if not failing or not failing <= all_fns.keys():
                sys.exit(f'probe link failed at P={P}:\n{res}')
            for f in failing:  # record the phase each failing function was linked at
                unlinkable[f].append(phase_of[f])
                if len(set(unlinkable[f])) == len(PHASES):
                    sys.exit(f'{f} does not link at any phase:\n{res}')
            excluded |= failing
        ins, syms = res
        brs = branches(ins)
        for f in all_fns:
            if f in excluded or f not in syms:
                continue
            start, size = syms[f]
            if start % 32 != P:
                sys.exit(f'{f} linked at phase {start % 32}, expected {P}')
            scores[f][P] = score(ins, brs, start, size)
        if verbose:
            print(f'phase {P:2d}: linked' + (f' (excluding {", ".join(sorted(excluded))})' if excluded else ''))
    return all_fns, scores, unlinkable


# --- main -------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--fix', action='store_true', help='rewrite the .S files with the best phases')
    ap.add_argument('--target', default='XK-EVK-XU416', help='VX4 target for xcc')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--build-dir', help='keep build files here (default: a temporary directory)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    work = Path(args.build_dir or tempfile.mkdtemp(prefix='vx4b_fetch_align_'))
    try:
        all_fns, scores, unlinkable = measure(LIB, work, args.target, args.jobs, args.verbose)
    finally:
        if not args.build_dir:
            shutil.rmtree(work, ignore_errors=True)

    committed = {}
    for rel in asm_sources(LIB):
        committed.update({f: p for f, p in phases_in(read(LIB / rel)[0]).items() if f in all_fns})

    problems, new_phase = [], {}
    for f in sorted(all_fns):
        sc = scores[f]
        best_w = min(w for w, _, _ in sc.values())
        best = min(sc, key=lambda P: (sc[P][0], P % 4 != 0, P))
        sensitive = len({w for w, _, _ in sc.values()}) > 1 or unlinkable[f]
        cur = committed.get(f)
        where = all_fns[f].as_posix()
        if cur is None:
            if sensitive:
                problems.append(f'{f} ({where}): not aligned; its loops are sensitive to their phase. '
                                f'Use `.p2align 5` + `.skip {best}` ({sc[best][1]} fetch stalls per iteration set)')
                new_phase[f] = best
        elif cur in unlinkable[f]:
            problems.append(f'{f} ({where}): `.skip {cur}` does not link (short branch out of range). Use .skip {best}')
            new_phase[f] = best
        elif cur not in sc:
            problems.append(f'{f} ({where}): `.skip {cur}` is not an even phase in 0..30. Use .skip {best}')
            new_phase[f] = best
        elif sc[cur][0] > best_w:
            problems.append(f'{f} ({where}): `.skip {cur}` gives {sc[cur][1]} fetch stalls per iteration set, '
                            f'`.skip {best}` gives {sc[best][1]}')
            new_phase[f] = best
        elif args.verbose:
            print(f'ok  {f}: P={cur} ({sc[cur][1]} stalls)')

    tuned = sum(1 for f in all_fns if committed.get(f) is not None)
    if not problems:
        print(f'vx4b fetch alignment OK: {tuned} of {len(all_fns)} functions have a tuned phase, all optimal')
        return 0
    print(f'vx4b fetch alignment: {len(problems)} problem(s)\n  ' + '\n  '.join(problems))
    if args.fix:
        for rel in asm_sources(LIB):
            text, eol = read(LIB / rel)
            sel = {f: new_phase[f] for f in functions(text) if f in new_phase}
            if sel:
                (LIB / rel).write_bytes(rewrite(text, sel).replace('\n', eol).encode())
                print(f'fixed {rel.as_posix()}')
        return 0
    print('Run `python lib_xcore_math/python/vx4b_fetch_align.py --fix` to update the .S files.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
