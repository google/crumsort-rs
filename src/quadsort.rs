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

use crate::Sortable;

fn swap_neighbours<T: Sortable>(slice: &mut [T], i: usize) -> bool {
    let first_greater = slice[i] > slice[i + 1];

    (slice[i], slice[i + 1]) = if first_greater {
        (slice[i + 1], slice[i])
    } else {
        (slice[i], slice[i + 1])
    };

    first_greater
}

fn unguarded_insert<T: Sortable>(slice: &mut [T], offset: usize) {
    assert!(offset <= slice.len());

    for i in offset..slice.len() {
        let mut j = i;

        if slice[j - 1] <= slice[j] {
            continue;
        }

        let key = slice[j];

        if slice[1] > key {
            slice.copy_within(1..i, 2);

            slice[1] = key;

            j = 0;
        } else {
            loop {
                (slice[j], slice[j - 1]) = (slice[j - 1], slice[j - 2]);

                j -= 2;

                if slice[j - 1] <= key {
                    break;
                }
            }

            (slice[j], slice[j + 1]) = (slice[j + 1], key);
        }

        swap_neighbours(slice, j);
    }
}

fn bubble_sort<T: Sortable>(slice: &mut [T]) {
    assert!(slice.len() <= 3);

    if slice.len() > 1 {
        if slice.len() > 2 {
            swap_neighbours(slice, 0);
            swap_neighbours(slice, 1);
        }

        swap_neighbours(slice, 0);
    }
}

fn quad_swap_four<T: Sortable>(slice: &mut [T; 4]) {
    swap_neighbours(slice, 0);
    swap_neighbours(slice, 2);

    if swap_neighbours(slice, 1) {
        swap_neighbours(slice, 0);
        swap_neighbours(slice, 2);
        swap_neighbours(slice, 1);
    }
}

#[derive(Debug)]
struct ParityMerger<const FROM_START: bool> {
    dst_i: usize,
    left_i: usize,
    right_i: usize,
}

impl ParityMerger<true> {
    pub const fn from_start() -> Self {
        Self {
            dst_i: 0,
            left_i: 0,
            right_i: 0,
        }
    }

    pub fn merge<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) {
        let left_less = left[self.left_i] <= right[self.right_i];
        let right_less = !left_less;

        dst[self.dst_i + usize::from(left_less)] = right[self.right_i];
        dst[self.dst_i + usize::from(right_less)] = left[self.left_i];

        self.left_i += usize::from(left_less);
        self.right_i += usize::from(right_less);
        self.dst_i += 1;
    }

    pub fn merge_last<T: Sortable>(&self, dst: &mut [T], left: &[T], right: &[T]) {
        dst[self.dst_i] = left[self.left_i].min(right[self.right_i]);
    }
}

impl ParityMerger<false> {
    pub const fn from_end(len: usize, half_len: usize) -> Self {
        Self {
            dst_i: len - 1,
            left_i: half_len - 1,
            right_i: half_len - 1,
        }
    }

    pub fn merge<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) {
        let left_greater = left[self.left_i] > right[self.right_i];
        let right_greater = !left_greater;

        dst[self.dst_i - usize::from(left_greater)] = right[self.right_i];
        dst[self.dst_i - usize::from(right_greater)] = left[self.left_i];

        self.left_i -= usize::from(left_greater);
        self.right_i -= usize::from(right_greater);
        self.dst_i -= 1;
    }

    pub fn merge_last<T: Sortable>(&self, dst: &mut [T], left: &[T], right: &[T]) {
        dst[self.dst_i] = left[self.left_i].max(right[self.right_i]);
    }
}

#[inline(always)]
fn parity_merge<T: Sortable>(dst: &mut [T], src: &[T]) {
    assert_eq!(dst.len(), src.len());

    let half_len = dst.len() / 2;
    let left = &src[..half_len];
    let right = &src[half_len..];

    let mut start_merger = ParityMerger::from_start();
    let mut end_merger = ParityMerger::from_end(dst.len(), half_len);

    for _ in 0..half_len - 1 {
        start_merger.merge(dst, left, right);
        end_merger.merge(dst, left, right);
    }

    start_merger.merge_last(dst, left, right);
    end_merger.merge_last(dst, left, right);
}

fn parity_swap_eight<T: Sortable>(slice: &mut [T; 8]) {
    let mut swap = [T::default(); 8];

    swap_neighbours(slice, 0);
    swap_neighbours(slice, 2);
    swap_neighbours(slice, 4);
    swap_neighbours(slice, 6);

    if slice[1] <= slice[2] && slice[3] <= slice[4] && slice[5] <= slice[6] {
        return;
    }

    parity_merge(&mut swap[..4], &slice[..4]);
    parity_merge(&mut swap[4..], &slice[4..]);

    parity_merge(slice, &swap);
}

fn parity_swap_sixteen<T: Sortable>(slice: &mut [T; 16]) {
    let mut swap = [T::default(); 16];

    quad_swap_four((&mut slice[0..4]).try_into().unwrap());
    quad_swap_four((&mut slice[4..8]).try_into().unwrap());
    quad_swap_four((&mut slice[8..12]).try_into().unwrap());
    quad_swap_four((&mut slice[12..16]).try_into().unwrap());

    if slice[3] <= slice[4] && slice[7] <= slice[8] && slice[11] <= slice[12] {
        return;
    }

    parity_merge(&mut swap[..8], &slice[..8]);
    parity_merge(&mut swap[8..], &slice[8..]);

    parity_merge(slice, &swap);
}

fn get_array<T: Sortable, const N: usize>(slice: &mut [T]) -> Option<&mut [T; N]> {
    slice.get_mut(..N).and_then(|slice| slice.try_into().ok())
}

pub(crate) fn tail_swap<T: Sortable>(slice: &mut [T]) {
    assert!(slice.len() < 32);

    if let Some(head) = get_array(slice) {
        parity_swap_sixteen(head);
        unguarded_insert(slice, 16);
    } else if let Some(head) = get_array(slice) {
        parity_swap_eight(head);
        unguarded_insert(slice, 8);
    } else if let Some(head) = get_array(slice) {
        quad_swap_four(head);
        unguarded_insert(slice, 4);
    } else {
        bubble_sort(slice);
    }
}

fn parity_tail_swap_eight<T: Sortable>(slice: &mut [T; 8]) {
    let mut swap = [T::default(); 8];

    swap_neighbours(slice, 4);

    if !swap_neighbours(slice, 6) && slice[3] <= slice[4] && slice[5] <= slice[6] {
        return;
    }

    swap[..4].copy_from_slice(&slice[..4]);
    parity_merge(&mut swap[4..], &slice[4..]);

    parity_merge(slice, &swap);
}

fn parity_tail_flip_eight<T: Sortable>(slice: &mut [T; 8]) {
    let mut swap = [T::default(); 8];

    if slice[3] <= slice[4] {
        return;
    }

    swap.copy_from_slice(slice);

    parity_merge(slice, &swap);
}

#[derive(Debug, Default)]
struct ForwardMerger {
    dst_i: usize,
    left_i: usize,
    right_i: usize,
}

impl ForwardMerger {
    fn copy_from<T: Sortable, const LEFT: bool>(&mut self, dst: &mut [T], src: &[T]) {
        let src_i = if LEFT {
            &mut self.left_i
        } else {
            &mut self.right_i
        };

        dst[self.dst_i] = src[*src_i];
        self.dst_i += 1;
        *src_i += 1;
    }

    pub fn merge_left<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) -> bool {
        let left_less = left[self.left_i + 1] <= right[self.right_i];

        if left_less {
            self.copy_from::<_, true>(dst, left);
            self.copy_from::<_, true>(dst, left);
        }

        left_less
    }

    pub fn merge_right<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) -> bool {
        let left_greater = left[self.left_i] > right[self.right_i + 1];

        if left_greater {
            self.copy_from::<_, false>(dst, right);
            self.copy_from::<_, false>(dst, right);
        }

        left_greater
    }

    pub fn merge<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) {
        let left_less = left[self.left_i] <= right[self.right_i];
        let right_less = !left_less;

        dst[self.dst_i + usize::from(left_less)] = right[self.right_i];
        dst[self.dst_i + usize::from(right_less)] = left[self.left_i];

        self.left_i += 1;
        self.right_i += 1;
        self.dst_i += 2;

        let left_less = left[self.left_i] <= right[self.right_i];
        let right_less = !left_less;

        dst[self.dst_i + usize::from(left_less)] = right[self.right_i];
        dst[self.dst_i + usize::from(right_less)] = left[self.left_i];

        self.left_i += usize::from(left_less);
        self.right_i += usize::from(right_less);
        self.dst_i += 1;
    }

    pub fn merge_last<T: Sortable>(&mut self, dst: &mut [T], left: &[T], right: &[T]) {
        let left_less = left[self.left_i] <= right[self.right_i];
        let right_less = !left_less;

        dst[self.dst_i] = left[self.left_i].min(right[self.right_i]);

        self.left_i += usize::from(left_less);
        self.right_i += usize::from(right_less);
        self.dst_i += 1;
    }
}
fn forward_merge<T: Sortable>(dst: &mut [T], src: &[T]) {
    assert!(dst.len() >= 4);
    assert_eq!(dst.len(), src.len());

    let half_len = dst.len() / 2;
    let left = &src[..half_len];
    let right = &src[half_len..];

    let mut merger = ForwardMerger::default();

    if left[half_len - 1] <= right[half_len - 1 - half_len / 4] {
        while merger.left_i < half_len - 2 {
            if !merger.merge_left(dst, left, right) && !merger.merge_right(dst, left, right) {
                merger.merge(dst, left, right);
            }
        }

        while merger.left_i < half_len {
            merger.merge_last(dst, left, right);
        }

        while merger.right_i < half_len {
            merger.copy_from::<_, false>(dst, right);
        }
    } else if left[half_len - 1 - half_len / 4] > right[half_len - 1] {
        while merger.right_i < half_len - 2 {
            if !merger.merge_right(dst, left, right) && !merger.merge_left(dst, left, right) {
                merger.merge(dst, left, right);
            }
        }

        while merger.right_i < half_len {
            merger.merge_last(dst, left, right);
        }

        while merger.left_i < half_len {
            merger.copy_from::<_, true>(dst, left);
        }
    } else {
        parity_merge(dst, src);
    }
}

fn quad_merge_block<T: Sortable>(slice: &mut [T], swap: &mut [T]) {
    assert!(slice.len() >= swap.len());

    let quarter_len = swap.len() / 4;

    if slice[quarter_len - 1] <= slice[quarter_len] {
        if slice[3 * quarter_len - 1] <= slice[3 * quarter_len] {
            if slice[2 * quarter_len - 1] <= slice[2 * quarter_len] {
                return;
            }

            swap.copy_from_slice(&slice[..4 * quarter_len]);

            return forward_merge(&mut slice[..4 * quarter_len], swap);
        }

        swap[..2 * quarter_len].copy_from_slice(&slice[..2 * quarter_len]);
    } else {
        forward_merge(&mut swap[..2 * quarter_len], &slice[..2 * quarter_len]);
    }

    forward_merge(
        &mut swap[2 * quarter_len..4 * quarter_len],
        &slice[2 * quarter_len..4 * quarter_len],
    );
    forward_merge(&mut slice[..4 * quarter_len], swap);
}

#[derive(Debug)]
struct PartialForwardMerger {
    swap_i: usize,
    start_i: usize,
    right_i: usize,
}

impl PartialForwardMerger {
    pub const fn new(block_len: usize) -> Self {
        Self {
            swap_i: 0,
            start_i: 0,
            right_i: block_len,
        }
    }

    pub fn right_to_start<T: Sortable>(&mut self, slice: &mut [T]) {
        slice[self.start_i] = slice[self.right_i];
        self.start_i += 1;
        self.right_i += 1;
    }

    pub fn swap_to_start<T: Sortable>(&mut self, slice: &mut [T], swap: &[T]) {
        slice[self.start_i] = swap[self.swap_i];
        self.swap_i += 1;
        self.start_i += 1;
    }

    pub fn merge<T: Sortable>(&mut self, slice: &mut [T], swap: &[T]) {
        if swap[self.swap_i] > slice[self.right_i + 1] {
            self.right_to_start(slice);
        } else if swap[self.swap_i + 1] <= slice[self.right_i] {
            self.swap_to_start(slice, swap);
        } else {
            let swap_less = swap[self.swap_i] <= slice[self.right_i];
            let right_less = !swap_less;

            slice[self.start_i + usize::from(swap_less)] = slice[self.right_i];
            slice[self.start_i + usize::from(right_less)] = swap[self.swap_i];

            self.swap_i += 1;
            self.start_i += 2;
            self.right_i += 1;

            let swap_less = swap[self.swap_i] <= slice[self.right_i];
            let right_less = !swap_less;

            slice[self.start_i + usize::from(swap_less)] = slice[self.right_i];
            slice[self.start_i + usize::from(right_less)] = swap[self.swap_i];

            self.swap_i += usize::from(swap_less);
            self.start_i += 1;
            self.right_i += usize::from(right_less);
        }
    }
}
fn partial_forward_merge<T: Sortable>(slice: &mut [T], mut swap: &mut [T], block_len: usize) {
    assert!(slice.len() > block_len);

    swap = &mut swap[..block_len];

    swap.copy_from_slice(&slice[..block_len]);

    let mut merger = PartialForwardMerger::new(block_len);
    while merger.swap_i < block_len - 2 && merger.right_i < slice.len() - 2 {
        merger.merge(slice, swap);
    }

    macro_rules! unwrap_or_break {
        ( $val:expr , $result:expr ) => {
            if let Some(val) = $val {
                val
            } else {
                break $result;
            }
        };
    }

    let needs_copy = loop {
        let swap_val = unwrap_or_break!(swap.get(merger.swap_i).copied(), false);
        let right_val = unwrap_or_break!(slice.get(merger.right_i).copied(), true);

        let swap_less = swap_val <= right_val;
        let right_less = !swap_less;

        slice[merger.start_i] = swap_val.min(right_val);

        merger.swap_i += usize::from(swap_less);
        merger.right_i += usize::from(right_less);
        merger.start_i += 1;
    };

    if needs_copy {
        let len = slice.len();
        slice[len - (swap.len() - merger.swap_i)..].copy_from_slice(&swap[merger.swap_i..]);
    }
}

#[derive(Debug)]
struct PartialBackwardMerger {
    swap_i: usize,
    mid_i: usize,
    end_i: usize,
}

impl PartialBackwardMerger {
    pub const fn new(len: usize, block_len: usize) -> Self {
        Self {
            swap_i: len - block_len - 1,
            mid_i: block_len - 1,
            end_i: len - 1,
        }
    }

    pub fn mid_to_end<T: Sortable>(&mut self, slice: &mut [T]) {
        slice[self.end_i] = slice[self.mid_i];
        self.mid_i -= 1;
        self.end_i -= 1;
    }

    pub fn swap_to_end<T: Sortable>(&mut self, slice: &mut [T], swap: &[T]) {
        slice[self.end_i] = swap[self.swap_i];
        self.swap_i -= 1;
        self.end_i -= 1;
    }

    pub fn merge<T: Sortable>(&mut self, slice: &mut [T], swap: &[T]) {
        if slice[self.mid_i - 1] > swap[self.swap_i] {
            self.mid_to_end(slice);
        } else if slice[self.mid_i] <= swap[self.swap_i - 1] {
            self.swap_to_end(slice, swap);
        } else {
            let mid_less = slice[self.mid_i] <= swap[self.swap_i];
            let swap_less = !mid_less;

            self.end_i -= 1;

            slice[self.end_i + usize::from(mid_less)] = swap[self.swap_i];
            slice[self.end_i + usize::from(swap_less)] = slice[self.mid_i];

            self.swap_i -= 1;
            self.mid_i -= 1;
            self.end_i -= 1;

            let mid_less = slice[self.mid_i] <= swap[self.swap_i];
            let swap_less = !mid_less;

            self.end_i -= 1;

            slice[self.end_i + usize::from(mid_less)] = swap[self.swap_i];
            slice[self.end_i + usize::from(swap_less)] = slice[self.mid_i];

            self.swap_i -= usize::from(mid_less);
            self.mid_i -= usize::from(swap_less);
        }
    }
}

fn partial_backward_merge<T: Sortable>(slice: &mut [T], swap: &mut [T], block_len: usize) {
    assert!(slice.len() > block_len);

    if slice[block_len - 1] <= slice[block_len] {
        return;
    }

    let len = slice.len();
    swap[..len - block_len].copy_from_slice(&slice[block_len..]);

    let mut merger = PartialBackwardMerger::new(len, block_len);
    while merger.swap_i > 1 && merger.mid_i > 1 {
        merger.merge(slice, swap);
    }

    macro_rules! sub_or_break {
        ( $lhs:expr , $rhs:expr , $result:expr ) => {
            if let Some(diff) = $lhs.checked_sub($rhs) {
                *$lhs = diff;
            } else {
                break $result;
            }
        };
    }

    let needs_copy = loop {
        let mid_greater = slice[merger.mid_i] > swap[merger.swap_i];
        let swap_greater = !mid_greater;

        slice[merger.end_i] = slice[merger.mid_i].max(swap[merger.swap_i]);

        sub_or_break!(&mut merger.swap_i, usize::from(swap_greater), false);
        sub_or_break!(&mut merger.mid_i, usize::from(mid_greater), true);

        merger.end_i -= 1;
    };

    if needs_copy {
        slice[..=merger.swap_i].copy_from_slice(&swap[..=merger.swap_i]);
    }
}

fn tail_merge<T: Sortable>(slice: &mut [T], swap: &mut [T], mut block_len: usize) {
    while block_len < slice.len() && block_len <= swap.len() {
        for chunk in slice.chunks_mut(2 * block_len) {
            if chunk.len() > block_len {
                partial_backward_merge(chunk, swap, block_len);
            }
        }

        block_len *= 2;
    }
}

fn quad_merge<T: Sortable>(slice: &mut [T], swap: &mut [T], mut block_len: usize) -> usize {
    assert!(block_len > 1);

    block_len *= 4;

    while block_len <= slice.len() && block_len <= swap.len() {
        let mut chunks = slice.chunks_exact_mut(block_len);

        for chunk in chunks.by_ref() {
            quad_merge_block(chunk, &mut swap[..block_len]);
        }

        tail_merge(chunks.into_remainder(), swap, block_len / 4);

        block_len *= 4;
    }

    tail_merge(slice, swap, block_len / 4);

    block_len / 2
}

fn monobound_binary_first<T: Sortable>(slice: &[T], val: T, mut top: usize) -> usize {
    let mut end = top;
    while top > 1 {
        let mid = top / 2;

        if val <= slice[end - mid] {
            end -= mid;
        }

        top -= mid;
    }

    if val <= slice[end - 1] {
        end -= 1;
    }

    end
}

fn blit_merge_block<T: Sortable>(
    slice: &mut [T],
    swap: &mut [T],
    mut left_block_len: usize,
    mut right: usize,
) {
    assert!(slice.len() > left_block_len);

    if slice[left_block_len - 1] <= slice[left_block_len] {
        return;
    }

    let right_block_len = left_block_len / 2;
    left_block_len -= right_block_len;

    let left = monobound_binary_first(
        &slice[left_block_len + right_block_len..],
        slice[left_block_len],
        right,
    );
    right -= left;

    if left > 0 {
        slice[left_block_len..left_block_len + right_block_len + left].rotate_left(right_block_len);

        if left <= swap.len() {
            partial_backward_merge(&mut slice[..left_block_len + left], swap, left_block_len);
        } else if left_block_len <= swap.len() {
            partial_forward_merge(&mut slice[..left_block_len + left], swap, left_block_len);
        } else {
            blit_merge_block(slice, swap, left_block_len, left);
        }
    }

    if right > 0 {
        if right <= swap.len() {
            partial_backward_merge(
                &mut slice[left_block_len + left..left_block_len + left + right_block_len + right],
                swap,
                right_block_len,
            );
        } else if right_block_len <= swap.len() {
            partial_forward_merge(
                &mut slice[left_block_len + left..left_block_len + left + right_block_len + right],
                swap,
                right_block_len,
            );
        } else {
            blit_merge_block(
                &mut slice[left_block_len + left..],
                swap,
                right_block_len,
                right,
            );
        }
    }
}

fn blit_merge<T: Sortable>(slice: &mut [T], swap: &mut [T], mut block_len: usize) {
    while block_len < slice.len() {
        for chunk in slice.chunks_mut(2 * block_len) {
            if chunk.len() > block_len {
                blit_merge_block(chunk, swap, block_len, chunk.len() - block_len);
            }
        }

        block_len *= 2;
    }
}

fn swap_four<T: Sortable>(slice: &mut [T; 4]) -> bool {
    match (
        slice[2] > slice[3],
        slice[1] > slice[2],
        slice[0] > slice[1],
    ) {
        (false, false, false) => false,
        (false, false, true) => {
            slice.swap(0, 1);

            swap_neighbours(slice, 1);
            swap_neighbours(slice, 2);

            false
        }
        (false, true, false) => {
            slice.swap(1, 2);

            swap_neighbours(slice, 0);
            swap_neighbours(slice, 2);
            swap_neighbours(slice, 1);

            false
        }
        (false, true, true) => {
            slice.swap(0, 2);

            swap_neighbours(slice, 2);
            swap_neighbours(slice, 1);

            false
        }
        (true, false, false) => {
            slice.swap(2, 3);

            swap_neighbours(slice, 1);
            swap_neighbours(slice, 0);

            false
        }
        (true, false, true) => {
            slice.swap(0, 1);
            slice.swap(2, 3);

            swap_neighbours(slice, 1);
            swap_neighbours(slice, 2);
            swap_neighbours(slice, 0);

            false
        }
        (true, true, false) => {
            slice.swap(1, 3);

            swap_neighbours(slice, 0);
            swap_neighbours(slice, 1);

            false
        }
        (true, true, true) => true,
    }
}

fn quad_swap<T: Sortable>(slice: &mut [T]) -> bool {
    let mut swap = [T::default(); 32];

    let mut i = 0;
    let mut count = slice.len() / 8 * 2;
    while count > 0 {
        count -= 1;

        if !swap_four((&mut slice[i..i + 4]).try_into().unwrap()) {
            count -= 1;

            parity_tail_swap_eight((&mut slice[i..i + 8]).try_into().unwrap());

            i += 8;

            continue;
        }

        let start_reverse = i;

        let count_not_zero = loop {
            i += 4;

            let new_count = count.checked_sub(1);

            if let Some(new_count) = new_count {
                count = new_count;

                if slice[i] > slice[i + 1] {
                    if slice[i + 2] > slice[i + 3] {
                        if slice[i + 1] > slice[i + 2] && slice[i - 1] > slice[i] {
                            continue;
                        }

                        slice.swap(i + 2, i + 3);
                    }

                    slice.swap(i, i + 1);
                } else if slice[i + 2] > slice[i + 3] {
                    slice.swap(i + 2, i + 3);
                }
            }

            break new_count.is_some();
        };

        if count_not_zero {
            if slice[i + 1] > slice[i + 2] {
                slice.swap(i + 1, i + 2);

                swap_neighbours(slice, i);
                swap_neighbours(slice, i + 2);
                swap_neighbours(slice, i + 1);
            }

            slice[start_reverse..i].reverse();

            if count % 2 == 0 {
                i -= 4;

                parity_tail_flip_eight((&mut slice[i..i + 8]).try_into().unwrap());
            } else {
                count -= 1;

                parity_tail_swap_eight((&mut slice[i..i + 8]).try_into().unwrap());
            }

            i += 8;

            continue;
        }

        if start_reverse == 0 {
            let is_reversed = !slice[i - 1..(i + 7).min(slice.len())]
                .windows(2)
                .any(|window| window[0] <= window[1]);

            if is_reversed {
                slice.reverse();

                return true;
            }
        }

        slice[start_reverse..i].reverse();

        break;
    }

    let len = slice.len();

    tail_swap(&mut slice[i..(i + 8).min(len)]);

    let mut chunks = slice.chunks_exact_mut(32);

    for chunk in chunks.by_ref() {
        if chunk[7] <= chunk[8] && chunk[15] <= chunk[16] && chunk[23] <= chunk[24] {
            continue;
        }

        parity_merge(&mut swap[..16], &chunk[..16]);
        parity_merge(&mut swap[16..], &chunk[16..]);

        parity_merge(chunk, &swap);
    }

    let tail = chunks.into_remainder();

    if !tail.is_empty() {
        tail_merge(tail, swap.as_mut_slice(), 8);
    }

    false
}

pub(crate) fn quadsort_swap<T: Sortable>(slice: &mut [T], swap: &mut [T]) {
    if slice.len() < 32 {
        tail_swap(slice);
    } else if !quad_swap(slice) {
        let block_len = quad_merge(slice, swap, 32);
        blit_merge(slice, swap, block_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unguarded_insert3() {
        let mut slice = [1, 2, 3, 4, 6, 5, 7];

        unguarded_insert(&mut slice, 4);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn unguarded_insert5() {
        let mut slice = [1, 2, 6, 3, 7, 5, 4];

        unguarded_insert(&mut slice, 2);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn bubble_sort_swapped() {
        let mut slice = [2, 1, 3];

        bubble_sort(&mut slice);

        assert_eq!(slice, [1, 2, 3]);
    }

    #[test]
    fn bubble_sort_reverse() {
        let mut slice = [3, 2, 1];

        bubble_sort(&mut slice);

        assert_eq!(slice, [1, 2, 3]);
    }

    #[test]
    fn quad_swap_four_swapped() {
        let mut slice = [2, 1, 4, 3];

        quad_swap_four(&mut slice);

        assert_eq!(slice, [1, 2, 3, 4]);
    }

    #[test]
    fn quad_swap_four_reverse() {
        let mut slice = [4, 3, 2, 1];

        quad_swap_four(&mut slice);

        assert_eq!(slice, [1, 2, 3, 4]);
    }

    #[test]
    fn parity_merge_zigzag() {
        let src = [1, 3, 5, 7, 2, 4, 6, 8];
        let mut dst = [0; 8];

        parity_merge(&mut dst, &src);

        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn parity_swap_eight_scrambled() {
        let mut slice = [3, 5, 2, 6, 7, 1, 8, 4];

        parity_swap_eight(&mut slice);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn parity_swap_sixteen_scrambled() {
        let mut slice = [3, 14, 13, 16, 6, 15, 4, 10, 7, 9, 12, 2, 11, 1, 8, 5];

        parity_swap_sixteen(&mut slice);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    fn tail_swap_scrambled(n: usize) {
        let mut slice = [
            6, 26, 20, 12, 9, 18, 21, 28, 17, 19, 24, 31, 14, 10, 30, 4, 25, 8, 5, 2, 29, 23, 15,
            1, 27, 3, 11, 7, 22, 16, 13,
        ];
        let mut clone = slice;

        tail_swap(&mut slice[..n]);
        clone[..n].sort();

        assert_eq!(slice[..n], clone[..n])
    }

    #[test]
    fn tail_swap_scrambled3() {
        tail_swap_scrambled(3);
    }

    #[test]
    fn tail_swap_scrambled7() {
        tail_swap_scrambled(7);
    }

    #[test]
    fn tail_swap_scrambled15() {
        tail_swap_scrambled(15);
    }

    #[test]
    fn tail_swap_scrambled31() {
        tail_swap_scrambled(31);
    }

    #[test]
    fn parity_tail_swap_eight_scrambled() {
        let mut slice = [1, 2, 3, 4, 7, 5, 8, 6];

        parity_tail_swap_eight(&mut slice);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn parity_tail_flip_eight_zigzag() {
        let mut slice = [1, 3, 5, 7, 2, 4, 6, 8];

        parity_tail_flip_eight(&mut slice);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn forward_merge_left() {
        let src = [1, 6, 7, 8, 2, 3, 4, 5];
        let mut dst = [0; 8];

        forward_merge(&mut dst, &src);

        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn forward_merge_right() {
        let src = [2, 3, 4, 5, 1, 6, 7, 8];
        let mut dst = [0; 8];

        forward_merge(&mut dst, &src);

        assert_eq!(dst, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn quad_merge_block_rotated2() {
        let mut slice = [3, 4, 5, 6, 7, 8, 1, 2];

        quad_merge_block(&mut slice, &mut [0; 8]);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn quad_merge_block_rotated4() {
        let mut slice = [5, 6, 7, 8, 1, 2, 3, 4];

        quad_merge_block(&mut slice, &mut [0; 8]);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn quad_merge_block_rotated6() {
        let mut slice = [7, 8, 1, 2, 3, 4, 5, 6];

        quad_merge_block(&mut slice, &mut [0; 8]);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn partial_forward_merge4() {
        let mut slice = [2, 3, 6, 8, 1, 4, 5, 7];

        partial_forward_merge(&mut slice, &mut [0; 4], 4);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn partial_backward_merge4() {
        let mut slice = [3, 4, 6, 8, 1, 2, 5, 7];

        partial_backward_merge(&mut slice, &mut [0; 4], 4);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn tail_merge2() {
        let mut slice = [4, 6, 1, 5, 2, 8, 3, 7];

        tail_merge(&mut slice, &mut [0; 4], 2);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn quad_merge_scrambled() {
        let mut slice = [
            5, 7, 14, 16, 9, 15, 3, 11, 6, 12, 1, 10, 8, 17, 4, 13, 18, 19, 2,
        ];

        assert_eq!(quad_merge(&mut slice, &mut [0; 8], 2), 16);

        assert_eq!(
            slice,
            [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 18, 19]
        );
    }

    #[test]
    fn monobound_binary_first_search() {
        let slice = [1, 2, 3, 4, 5, 6, 7, 8];

        for val in slice {
            assert_eq!(monobound_binary_first(&slice, val, slice.len()), val - 1);
        }
    }

    #[test]
    fn blit_merge_block4_left() {
        let mut slice = [2, 4, 5, 8, 1, 3, 6, 7];

        blit_merge_block(&mut slice, &mut [0; 4], 4, 4);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn blit_merge_block8_left() {
        let mut slice = [2, 3, 4, 12, 13, 14, 15, 16, 1, 5, 6, 7, 8, 9, 10, 11];

        blit_merge_block(&mut slice, &mut [0; 4], 8, 8);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn blit_merge_block10_left() {
        let mut slice = [2, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 3, 4, 5, 6, 7];

        blit_merge_block(&mut slice, &mut [0; 4], 10, 6);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn blit_merge_block4_right() {
        let mut slice = [1, 4, 5, 8, 2, 3, 6, 7];

        blit_merge_block(&mut slice, &mut [0; 4], 4, 4);

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn blit_merge_block8_right() {
        let mut slice = [1, 3, 4, 5, 6, 9, 10, 11, 2, 7, 8, 12, 13, 14, 15, 16];

        blit_merge_block(&mut slice, &mut [0; 4], 8, 8);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn blit_merge_block10_right() {
        let mut slice = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 8, 12, 13, 14, 15, 16];

        blit_merge_block(&mut slice, &mut [0; 4], 10, 6);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        );
    }

    #[test]
    fn blit_merge_scrambled() {
        let mut slice = [
            5, 7, 14, 16, 3, 9, 11, 15, 1, 6, 10, 12, 4, 8, 13, 17, 2, 18, 19,
        ];

        blit_merge(&mut slice, &mut [0; 8], 4);

        assert_eq!(
            slice,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn quad_swap_all_swaps() {
        let mut slice = [
            5, 6, 8, 7, 9, 11, 10, 12, 13, 16, 15, 14, 18, 17, 19, 20, 22, 21, 24, 23, 27, 26, 25,
            28, 1, 2, 3, 4,
        ];

        assert!(!quad_swap(&mut slice));

        assert_eq!(
            slice,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28,
            ],
        );
    }

    #[test]
    fn quad_swap_reverse8() {
        let mut slice = [8, 7, 6, 5, 3, 2, 1, 4];

        assert!(!quad_swap(&mut slice));

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn quad_swap_reverse12() {
        let mut slice = [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 4];

        assert!(!quad_swap(&mut slice));

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn quad_swap_reverse_all() {
        let mut slice = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        assert!(quad_swap(&mut slice));

        assert_eq!(slice, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }
}
