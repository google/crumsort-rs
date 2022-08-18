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

#![deny(missing_docs)]

//! # crumsort
//!
//! A parallelized Rust port of [crumsort](https://github.com/scandum/crumsort).
//!
//! # Usage
//!
//! ```
//! use crumsort::ParCrumSort;
//!
//! let mut vals = [5, 4, 1, 3, 2];
//!
//! vals.par_crumsort();
//!
//! assert_eq!(vals, [1, 2, 3, 4, 5]);
//! ```

mod quadsort;

const SWAP_SIZE: usize = 512;
const CRUM_CHUNK_SIZE: usize = 32;
const CRUM_OUT: usize = 24;
const JOIN_THRESHOLD: usize = 1_024;

#[allow(clippy::assertions_on_constants)]
const _: () = assert!(2 * CRUM_CHUNK_SIZE <= SWAP_SIZE);

pub(crate) trait Sortable: Copy + Default + Ord {}

impl<T: Copy + Default + Ord> Sortable for T {}

fn crum_median_of_sqrt<T: Sortable>(slice: &mut [T], swap: &mut [T]) -> usize {
    let sqrt = match slice.len() {
        0..=65_535 => 16,
        65_536..=262_143 => 128,
        _ => 256,
    };

    let div = slice.len() / sqrt;
    let mut end = slice.len() - 1;

    for i in (0..sqrt).rev() {
        slice.swap(i, end);

        end = end.saturating_sub(div);
    }

    quadsort::quadsort_swap(&mut slice[..sqrt], swap);

    sqrt / 2
}

fn crum_median_of_three<T: Sortable>(slice: &[T], indices: [usize; 3]) -> usize {
    let x = u8::from(slice[indices[0]] > slice[indices[1]]);
    let y = u8::from(slice[indices[0]] > slice[indices[2]]);
    let z = u8::from(slice[indices[1]] > slice[indices[2]]);

    indices[usize::from(u8::from(x == y) + (y ^ z))]
}

fn crum_median_of_nine<T: Sortable>(slice: &[T]) -> usize {
    let div = slice.len() / 16;

    let x = crum_median_of_three(slice, [div * 2, div, div * 4]);
    let y = crum_median_of_three(slice, [div * 8, div * 6, div * 10]);
    let z = crum_median_of_three(slice, [div * 14, div * 12, div * 15]);

    crum_median_of_three(slice, [x, y, z])
}

#[derive(Debug)]
struct Partitioner {
    left_i: usize,
    right_i: usize,
    end_i: usize,
    cursor: usize,
}

impl Partitioner {
    pub const fn new<const C: usize>(len: usize) -> Self {
        Self {
            left_i: C,
            right_i: len - 1 - C,
            end_i: len - 1,
            cursor: 0,
        }
    }

    pub fn next<T: Sortable, const IS_LEFT: bool>(
        &mut self,
        slice: &mut [T],
        swap: Option<&mut [T]>,
        pivot: T,
    ) {
        let val = {
            let i = if IS_LEFT { self.left_i } else { self.right_i };
            let val = swap.map_or_else(
                || {
                    // SAFETY:
                    // `Partitioner::next` is called `slice.len()` times ==> 0 <= i < `slice.len()`
                    unsafe { *slice.get_unchecked(i) }
                },
                |swap| {
                    // SAFETY:
                    // `Partitioner::next` is called `slice.len()` times ==> 0 <= i < `slice.len()` (1)
                    // i < 2 * CRUM_CHUNK_SIZE (2)
                    // 2 * CRUM_CHUNK_SIZE <= `swap.len()` (3)
                    // (1), (2), (3) ==> 0 <= i < `swap.len()`
                    unsafe { *swap.get_unchecked(i) }
                },
            );

            // SAFETY:
            // `usize::from(val <= pivot)` <= 1, `Partitioner::next` is called `slice.len()`
            // times ==> self.cursor <= i` (1)
            // (1), i < `slice.len()` ==> self.cursor < `slice.len()`
            unsafe {
                *slice.get_unchecked_mut(self.cursor) = val;
            }

            // SAFETY:
            // for nth iteration:
            //   `usize::from(val <= pivot)` <= 1 ==> self.cursor <= n (1)
            //   self.end_i = `slice.len()` - 1 - n (2)
            //   (1), (2) ==> self.cursor + self.end_i <= `slice.len()` - 1 <==>
            //   <==> self.cursor + self.end_i < `slice.len()`
            unsafe {
                *slice.get_unchecked_mut(self.cursor + self.end_i) = val;
            }

            val
        };

        if IS_LEFT {
            self.left_i += 1;
        } else {
            self.right_i -= 1;
        }

        self.end_i = self.end_i.overflowing_sub(1).0;
        self.cursor += usize::from(val <= pivot);
    }
}

fn fulcrum_partition_inner<T: Sortable>(slice: &mut [T], swap: &mut [T], pivot: T) -> usize {
    if slice.len() <= swap.len() {
        const CHUNK: usize = 8;

        let mut i = 0;
        let mut cursor = 0;

        let mut partition = |slice: &mut [T]| {
            let val = {
                // SAFETY:
                // `partition` is called `slice.len()` times ==> i < `slice.len()`
                let val = unsafe { *slice.get_unchecked(i) };

                // SAFETY:
                // `swap.len` >= `slice.len()` ==> i < `swap.len()` (1)
                // `usize::from(val <= pivot)` <= 1, `partition` is called `slice.len()` times ==>
                // => cursor <= i (2)
                // (1), (2) ==> 0 <= i - cursor < `swap.len()`
                unsafe {
                    *swap.get_unchecked_mut(i - cursor) = val;
                }

                // SAFETY:
                // cursor <= i, i < `slice.len()` ==> cursor < `slice.len()`
                unsafe {
                    *slice.get_unchecked_mut(cursor) = val;
                }

                val
            };

            i += 1;
            cursor += usize::from(val <= pivot);
        };

        for _ in 0..slice.len() / CHUNK {
            for _ in 0..CHUNK {
                partition(slice);
            }
        }

        for _ in 0..slice.len() % CHUNK {
            partition(slice);
        }

        let len = slice.len();
        slice[cursor..].copy_from_slice(&swap[..len - cursor]);

        return cursor;
    }

    swap[..CRUM_CHUNK_SIZE].copy_from_slice(&slice[..CRUM_CHUNK_SIZE]);
    swap[CRUM_CHUNK_SIZE..2 * CRUM_CHUNK_SIZE]
        .copy_from_slice(&slice[slice.len() - CRUM_CHUNK_SIZE..]);

    let mut partitioner = Partitioner::new::<CRUM_CHUNK_SIZE>(slice.len());

    let mut count = slice.len() / CRUM_CHUNK_SIZE - 2;

    loop {
        if partitioner.left_i - partitioner.cursor <= CRUM_CHUNK_SIZE {
            if let Some(new_count) = count.checked_sub(1) {
                count = new_count;
            } else {
                break;
            }

            for _ in 0..CRUM_CHUNK_SIZE {
                partitioner.next::<_, true>(slice, None, pivot);
            }
        }

        if partitioner.left_i - partitioner.cursor > CRUM_CHUNK_SIZE {
            if let Some(new_count) = count.checked_sub(1) {
                count = new_count;
            } else {
                break;
            }

            for _ in 0..CRUM_CHUNK_SIZE {
                partitioner.next::<_, false>(slice, None, pivot);
            }
        }
    }

    if partitioner.left_i - partitioner.cursor <= CRUM_CHUNK_SIZE {
        for _ in 0..slice.len() % CRUM_CHUNK_SIZE {
            partitioner.next::<_, true>(slice, None, pivot);
        }
    } else {
        for _ in 0..slice.len() % CRUM_CHUNK_SIZE {
            partitioner.next::<_, false>(slice, None, pivot);
        }
    }

    partitioner.left_i = 0;
    for _ in 0..2 * CRUM_CHUNK_SIZE {
        partitioner.next::<_, true>(slice, Some(swap), pivot);
    }

    partitioner.cursor
}

fn fulcrum_partition<T: Sortable + Send>(slice: &mut [T], swap: &mut [T], max: Option<T>) {
    let i = if slice.len() <= 2_048 {
        crum_median_of_nine(slice)
    } else {
        crum_median_of_sqrt(slice, swap)
    };

    let pivot = slice[i];

    if let Some(max_val) = max {
        if max_val <= pivot {
            let left = fulcrum_partition_inner(slice, swap, pivot);
            let right = slice.len() - left;

            if right <= left / 16 || left <= CRUM_OUT {
                return quadsort::quadsort_swap(&mut slice[..left], swap);
            }

            return fulcrum_partition(&mut slice[..left], swap, None);
        }
    }

    let len = slice.len() - 1;

    slice[i] = slice[len];

    let left_i = fulcrum_partition_inner(&mut slice[..len], swap, pivot);
    let right_i = len - left_i;

    slice[len] = slice[left_i];
    slice[left_i] = pivot;

    if left_i <= right_i / 16 || right_i <= CRUM_OUT {
        if right_i == 0 {
            let left = fulcrum_partition_inner(&mut slice[..left_i], swap, pivot);
            let right = len - left;

            if right <= left / 16 || left <= CRUM_OUT {
                return quadsort::quadsort_swap(&mut slice[..left], swap);
            }

            return fulcrum_partition(&mut slice[..left], swap, None);
        }

        quadsort::quadsort_swap(&mut slice[left_i + 1..], swap);
    }

    if slice.len() > JOIN_THRESHOLD {
        let (left, right) = slice.split_at_mut(left_i + 1);

        rayon::join(
            move || {
                if !(left_i <= right_i / 16 || right_i <= CRUM_OUT) {
                    fulcrum_partition(right, swap, max);
                }
            },
            || {
                let mut swap = [T::default(); SWAP_SIZE];

                if right_i <= left_i / 32 || left_i <= CRUM_OUT {
                    return quadsort::quadsort_swap(left, &mut swap);
                }

                let max = Some(left[left.len() - 1]);
                fulcrum_partition(left, &mut swap, max);
            },
        );
    } else {
        if !(left_i <= right_i / 16 || right_i <= CRUM_OUT) {
            fulcrum_partition(&mut slice[left_i + 1..], swap, max);
        }

        if right_i <= left_i / 32 || left_i <= CRUM_OUT {
            return quadsort::quadsort_swap(&mut slice[..left_i], swap);
        }

        let max = Some(slice[left_i]);
        fulcrum_partition(&mut slice[..left_i], swap, max);
    }
}

/// Parallel sort extension trait.
pub trait ParCrumSort {
    /// Unstably sorts the slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use crumsort::ParCrumSort;
    ///
    /// let mut vals = [5, 4, 1, 3, 2];
    ///
    /// vals.par_crumsort();
    ///
    /// assert_eq!(vals, [1, 2, 3, 4, 5]);
    /// ```
    fn par_crumsort(&mut self);
}

impl<T: Copy + Default + Ord + Send> ParCrumSort for [T] {
    fn par_crumsort(&mut self) {
        if self.len() < 32 {
            return quadsort::tail_swap(self);
        }

        fulcrum_partition(self, &mut [T::default(); SWAP_SIZE], None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::prelude::*;

    fn test_partition<const N: usize>(thirds: usize) {
        let mut slice: Vec<_> = (0..N).into_iter().map(|i| N - i - 1).collect();

        let pivot = slice[thirds * slice.len() / 3];
        fulcrum_partition_inner(&mut slice, &mut [0; 2 * CRUM_CHUNK_SIZE], pivot);

        let (left, right) = slice.split_at(pivot + 1);

        assert!(left.iter().all(|&val| val <= pivot));
        assert!(right.iter().all(|&val| val > pivot));
    }

    #[test]
    fn partition7() {
        test_partition::<7>(1);
    }

    #[test]
    fn partition15() {
        test_partition::<15>(1);
    }

    #[test]
    fn partition31() {
        test_partition::<31>(1);
    }

    #[test]
    fn partition159() {
        test_partition::<159>(1);
    }

    #[test]
    fn partition193() {
        test_partition::<193>(2);
    }

    #[test]
    fn sort_16_384_uniform() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vals: Vec<u32> = (0..16_384).into_iter().map(|_| rng.gen()).collect();
        let mut sorted = vals.clone();

        sorted.sort_unstable();

        vals.as_mut_slice().par_crumsort();

        assert_eq!(vals, sorted);
    }

    #[test]
    fn sort_16_384_uniform_small_range() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vals: Vec<u32> = (0..16_384)
            .into_iter()
            .map(|_| rng.gen_range(0..64))
            .collect();
        let mut sorted = vals.clone();

        sorted.sort_unstable();

        vals.as_mut_slice().par_crumsort();

        assert_eq!(vals, sorted);
    }
}
