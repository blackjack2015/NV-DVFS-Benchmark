/**
 * mix_kernels_cuda.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#pragma once

enum BenchType
{
    BENCH_INT,
    BENCH_FLOAT,
    BENCH_DOUBLE
};

extern "C" void mixbenchGPU(double*, long size, int, BenchType, int);


