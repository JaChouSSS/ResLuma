#!/usr/bin/env python3
"""Benchmark the two curve-only operators from ResLuma manuscript (Nov 4, 2026).

No model, image inference, global exposure statistics, clipping or local compensation.
These are *new local measurements* on synthetic inputs, not a way to recover the
original author's reported milliseconds exactly. CPU hardware matters greatly.

Usage:
    python benchmarks/benchmark_cpu_curves.py
    python benchmarks/benchmark_cpu_curves.py --all --out measured_cpu_timing.csv
Requires PyTorch installed separately, e.g. matching your machine environment.
"""
import argparse, csv, time
import torch

C0,C1=0.5796,0.9460

def iterative_eight(x,s):
    y=x
    for _ in range(8):
        y=y+s*y*(1.0-y)
    return y

def high_order_corrected(x,s):
    a=1.0+s
    a=a*a
    a=a*a
    a=a*a
    ax=a*x
    r=ax/(1.0-x+ax+1e-6)
    return r+r*(1.0-r)*s*(C0+C1*(2.0*r-1.0))

def run_once(fn,x,s):
    t=time.perf_counter()
    y=fn(x,s)
    return (time.perf_counter()-t)*1000.0, y

def measure(fn,x,s):
    with torch.no_grad():
        for _ in range(3): run_once(fn,x,s)
        return min(run_once(fn,x,s)[0] for _ in range(8))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--all',action='store_true',help='measure all six sizes, including memory-heavy 8K')
    p.add_argument('--out',default='measured_cpu_timing.csv')
    args=p.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    sizes=[(512,512)]
    if args.all: sizes=[(512,512),(1280,720),(1920,1080),(2560,1440),(3840,2160),(7680,4320)]
    rows=[]
    for w,h in sizes:
        print(f'Measuring {w}x{h}, 1x3xHxW FP32, eight CPU threads ...',flush=True)
        x=torch.rand((1,3,h,w),dtype=torch.float32)
        s=torch.rand((1,1,h,w),dtype=torch.float32)*0.5
        a=measure(iterative_eight,x,s)
        b=measure(high_order_corrected,x,s)
        rows.append((f'{w}x{h}',round(a,3),round(b,3),round(a/b,3)))
        del x,s
    with open(args.out,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['resolution','eight_step_iteration_ms_measured','high_order_mapping_ms_measured','measured_speedup'])
        writer.writerows(rows)
    print(f'Saved {args.out}. Local timings are hardware-dependent, not original paper Table 6 data.')

if __name__=='__main__': main()
