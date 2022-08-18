// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{fmt, time::Instant};


use crumsort::ParCrumSort;
use rand::prelude::*;
use rayon::slice::ParallelSliceMut;
use tabled::{object::Columns, Alignment, Modify, Style, Table, Tabled};

const ALGOS: &[Algo] = &[Algo::PdqSort, Algo::CrumSort];
const SIZES: &[usize] = &[1, 4, 8, 16];
const LENS: &[usize] = &[1 << 12, 1 << 16, 1 << 20, 1 << 24];

#[derive(Clone, Copy, Debug, Tabled)]
enum Algo {
    PdqSort,
    CrumSort,
}

impl fmt::Display for Algo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Algo::PdqSort => "pdqsort",
            Algo::CrumSort => "crumsort",
        };

        write!(f, "{}", name)
    }
}

impl Algo {
    pub fn sort<T: Copy + Default + Ord + Send + Sync>(self, slice: &mut [T]) {
        match self {
            Algo::PdqSort => slice.par_sort_unstable(),
            Algo::CrumSort => slice.par_crumsort(),
        }
    }
}

fn format_f32(val: f32) -> String {
    if val < 1e3 {
        format!("{:.2}", val)
    } else if val < 1e6 {
        format!("{:.2}K", val / 1e3)
    } else if val < 1e9 {
        format!("{:.2}M", val / 1e6)
    } else {
        format!("{:.2}G", val / 1e9)
    }
}

fn display_throughput(throughput: &f32) -> String {
    format!("{}keys/s", format_f32(*throughput))
}

fn display_improvement(improvement: &f32) -> String {
    format!("{:.2}%", *improvement * 100.0)
}

#[derive(Debug, Tabled)]
struct Entry {
    length: usize,
    algorithm: Algo,
    #[tabled(display_with = "display_throughput")]
    throughput: f32,
    #[tabled(display_with = "display_improvement")]
    improvement: f32,
}

fn measure<F: FnMut()>(mut f: F) -> f64 {
    let now = Instant::now();

    f();

    now.elapsed().as_secs_f64()
}

fn main() {
    println!("\n");

    let mut rng = StdRng::seed_from_u64(42);

    for &size in SIZES {
        let mut entries = Vec::new();

        for &len in LENS {
            for &algo in ALGOS {
                let mut timings = Vec::new();

                for _ in 0..((1 << 28) / len).min(1 << 14) {
                    match size {
                        1 => {
                            let mut vals: Vec<u8> =
                                (0..len).into_iter().map(|_| rng.gen()).collect();

                            timings.push(measure(|| algo.sort(&mut vals)));
                        }
                        4 => {
                            let mut vals: Vec<u32> =
                                (0..len).into_iter().map(|_| rng.gen()).collect();

                            timings.push(measure(|| algo.sort(&mut vals)));
                        }
                        8 => {
                            let mut vals: Vec<u64> =
                                (0..len).into_iter().map(|_| rng.gen()).collect();

                            timings.push(measure(|| algo.sort(&mut vals)));
                        }
                        16 => {
                            let mut vals: Vec<u64> =
                                (0..len).into_iter().map(|_| rng.gen()).collect();

                            timings.push(measure(|| algo.sort(&mut vals)));
                        }
                        _ => unreachable!(),
                    }
                }

                let average = timings.iter().copied().sum::<f64>() / timings.len() as f64;
                let throughput = (len as f64 / average) as f32;

                entries.push(Entry {
                    algorithm: algo,
                    length: len,
                    throughput,
                    improvement: 0.0,
                });
            }

            let reference = entries
                .iter()
                .rev()
                .take(ALGOS.len())
                .last()
                .unwrap()
                .throughput;

            for entry in entries.iter_mut().rev().take(ALGOS.len()) {
                entry.improvement = entry.throughput / reference - 1.0;
            }
        }

        let style = Style::modern()
            .off_horizontal()
            .lines([(1, Style::modern().get_horizontal().horizontal(Some('═')))]);
        let table = Table::new(entries)
            .with(style)
            .with(Modify::new(Columns::single(0)).with(Alignment::right()))
            .with(Modify::new(Columns::single(1)).with(Alignment::center()))
            .with(Modify::new(Columns::single(2)).with(Alignment::right()))
            .with(Modify::new(Columns::single(3)).with(Alignment::right()));

        println!("u{}\n{}\n\n", 8 * size, table.to_string());
    }
}
