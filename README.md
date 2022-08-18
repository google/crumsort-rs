# crumsort-rs

A parallelized Rust port of [crumsort](https://github.com/scandum/crumsort).

The goal of this port is to excel at sorting well-distributed data which is why it is not an exact 1:1 replica.

## Temporary caveats

* sorts uniform data faster than crumsort, but severly skewed distributions slower
* intended as a solution for sorting large (`u64` or `u128`) integers
* only sorts `Copy + Default` data as a way to limit the use of `unsafe`
* missing un-parallelized version (data needs to implement `Send`)
* missing `*_by` and `*_by_key` sorting alternatives (data needs to implement `Ord`)

## Benchmarks against parallel pdqsort (Rayon)

All banchmarks run with the `bench` example on an M1 Pro.

### Uniformly distributed random `u32`s

|     Length     | Algorithm |    Throughput | Improvement |
|:--------------:|:---------:|--------------:|------------:|
| 2<sup>12</sup> |  pdqsort  |  32.15Mkeys/s |       0.00% |
| 2<sup>12</sup> | crumsort  |  38.70Mkeys/s |      20.39% |
| 2<sup>16</sup> |  pdqsort  | 129.96Mkeys/s |       0.00% |
| 2<sup>16</sup> | crumsort  | 176.95Mkeys/s |      36.16% |
| 2<sup>20</sup> |  pdqsort  | 226.31Mkeys/s |       0.00% |
| 2<sup>20</sup> | crumsort  | 368.09Mkeys/s |      62.65% |
| 2<sup>24</sup> |  pdqsort  | 227.80Mkeys/s |       0.00% |
| 2<sup>24</sup> | crumsort  | 399.89Mkeys/s |      75.54% |

### Uniformly distributed random `u64`s

|     Length     | Algorithm |    Throughput | Improvement |
|:--------------:|:---------:|--------------:|------------:|
| 2<sup>12</sup> |  pdqsort  |  33.18Mkeys/s |       0.00% |
| 2<sup>12</sup> | crumsort  |  40.79Mkeys/s |      22.91% |
| 2<sup>16</sup> |  pdqsort  | 151.24Mkeys/s |       0.00% |
| 2<sup>16</sup> | crumsort  | 237.48Mkeys/s |      57.02% |
| 2<sup>20</sup> |  pdqsort  | 218.64Mkeys/s |       0.00% |
| 2<sup>20</sup> | crumsort  | 364.79Mkeys/s |      66.85% |
| 2<sup>24</sup> |  pdqsort  | 226.83Mkeys/s |       0.00% |
| 2<sup>24</sup> | crumsort  | 385.42Mkeys/s |      69.92% |
