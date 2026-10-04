use crate::{Blueprint, CubeScanStrategy, GlobalScanStrategy, instructions::ScanInstruction};
use cubecl::prelude::*;

#[cube]
fn plane_inclusive_prefix_scan<T: Numeric, I: ScanInstruction>(
    mut value: T,
    #[comptime] blueprint: &Blueprint,
) -> T {
    // Kogge-Stone / Hillis-Steele
    #[unroll]
    for shift in 0..blueprint.plane_size_log {
        let offset = (1usize << shift) as u32;
        let below = plane_shuffle(value, UNIT_POS_PLANE - offset);
        if UNIT_POS_PLANE >= offset {
            value = I::combine::<T>(below, value);
        }
    }
    value
}

#[cube(launch_unchecked)]
pub(crate) fn single_pass_inclusive_prefix_scan<T: Numeric, N: Size, I: ScanInstruction>(
    input: &Tensor<Vector<T, N>>,
    output: &mut Tensor<Vector<T, N>>,
    output_atomic: &mut Tensor<Atomic<T>>,
    task_allocator: &mut Tensor<Atomic<u32>>,
    task_status: &mut Tensor<Atomic<u32>>,
    #[comptime] blueprint: Blueprint,
    #[define(T)] _dtype: ElemType,
) {
    let mut cube_id = CUBE_POS_X as usize;
    #[comptime]
    if blueprint.use_task_allocator {
        let mut cube_broadcast = Shared::<u32>::new();
        if UNIT_POS_X == 0 {
            *cube_broadcast = task_allocator[0].fetch_add(1u32);
        }
        sync_cube();
        cube_id = *cube_broadcast as usize;
    }

    // Vectorized load from global memory (parallel)
    let mut vector = input[cube_id * blueprint.cube_size + UNIT_POS_X as usize];

    // Unit scan (sequential)
    #[unroll]
    for v in 1..input.vector_size() {
        vector.insert(v, I::combine::<T>(vector.extract(v - 1), vector.extract(v)));
    }

    // Cube scan (parallel)
    let mut cube_scan = Shared::new_slice(blueprint.cube_size);
    cube_scan[UNIT_POS_X as usize] = vector.extract(input.vector_size().comptime() - 1);
    sync_cube();
    #[comptime]
    match blueprint.cube_scan_strategy {
        CubeScanStrategy::KoggeStoneOrHillisSteele =>
        {
            #[unroll]
            for shift in 0..blueprint.cube_size_log {
                let offset = 1u32 << shift;
                let below = select(
                    UNIT_POS_X >= offset,
                    cube_scan[(UNIT_POS_X - offset) as usize],
                    I::identity::<T>(),
                );
                sync_cube();
                cube_scan[UNIT_POS_X as usize] =
                    I::combine::<T>(below, cube_scan[UNIT_POS_X as usize]);
                sync_cube();
            }
        }
        CubeScanStrategy::Blelloch => {
            // Up-sweep phase: Reduction
            #[unroll]
            for shift in 0..blueprint.cube_size_log {
                let offset = 1u32 << shift;
                let modulo_mask = (2u32 << shift) - 1;
                let active_residue_class = modulo_mask;
                let is_unit_active = UNIT_POS_X & modulo_mask == active_residue_class;
                let mut below = I::identity::<T>();
                if is_unit_active {
                    below = cube_scan[(UNIT_POS_X - offset) as usize];
                }
                sync_cube();
                if is_unit_active {
                    cube_scan[UNIT_POS_X as usize] =
                        I::combine::<T>(below, cube_scan[UNIT_POS_X as usize]);
                }
                sync_cube();
            }
            // Down-sweep phase: Inclusive prefix scan
            // Note: This differs from the paper which does an exclusive scan.
            #[unroll]
            for shift in 1..blueprint.cube_size_log {
                let offset = blueprint.cube_size as u32 >> (shift + 1);
                let modulo_mask = (blueprint.cube_size as u32 >> shift) - 1;
                let active_residue_class = offset - 1;
                let is_unit_active =
                    (UNIT_POS_X & modulo_mask == active_residue_class) & (UNIT_POS_X >= offset);
                let mut below = I::identity::<T>();
                if is_unit_active {
                    below = cube_scan[(UNIT_POS_X - offset) as usize];
                }
                sync_cube();
                if is_unit_active {
                    cube_scan[UNIT_POS_X as usize] =
                        I::combine::<T>(below, cube_scan[UNIT_POS_X as usize]);
                }
                sync_cube();
            }
        }
        CubeScanStrategy::BrentKung => {
            // Up-sweep phase: Inclusive prefix scan
            #[unroll]
            for layer in 0..blueprint.plane_sized_layer_count {
                let scale = 1 << (layer * blueprint.plane_size_log);
                let index = UNIT_POS_X as usize * scale + scale - 1;
                let plane_scan = plane_inclusive_prefix_scan::<T, I>(
                    select(
                        index < blueprint.cube_size,
                        cube_scan[index],
                        I::identity::<T>(),
                    ),
                    &blueprint,
                );
                if index < blueprint.cube_size {
                    cube_scan[index] = plane_scan;
                }
                sync_cube();
            }
            // Down-sweep phase: Propagation
            #[unroll]
            for layer in 0..blueprint.plane_sized_layer_count {
                let reverse_layer = blueprint.plane_sized_layer_count - layer - 1;
                let scale = 1 << (reverse_layer * blueprint.plane_size_log);
                let index = UNIT_POS_X as usize * scale + scale - 1;
                if PLANE_POS > 0
                    && (UNIT_POS_PLANE as usize) < blueprint.plane_size - 1
                    && index < blueprint.cube_size
                {
                    let base = cube_scan
                        [(PLANE_POS as usize * blueprint.plane_size - 1) * scale + scale - 1];
                    cube_scan[index] += base;
                }
                sync_cube();
            }
        }
    }

    // Global scan
    if UNIT_POS_X == 0 {
        #[comptime]
        match blueprint.global_scan_strategy {
            GlobalScanStrategy::StreamScan => {
                let mut spin_cycles = 1u32;
                let mut cube_prefix = I::identity::<T>();
                if cube_id > 0 {
                    while task_status[cube_id - 1].load() == 0 {
                        spin_cycles += 1;
                    }
                    cube_prefix = output_atomic
                        [cube_id * blueprint.cube_size * input.vector_size() - 1]
                        .load();
                }

                // Global broadcast
                let cube_reduction = cube_scan[blueprint.cube_size - 1];
                output_atomic[(cube_id + 1) * blueprint.cube_size * input.vector_size() - 1]
                    .store(I::combine::<T>(cube_prefix, cube_reduction));
                task_status[cube_id].store(spin_cycles);

                cube_scan[blueprint.cube_size - 1] = cube_prefix;
            }
        }
    }
    sync_cube();

    // Unit scan (continued)
    let cube_prefix = cube_scan[blueprint.cube_size - 1];
    let unit_prefix = select(
        UNIT_POS_X > 0,
        cube_scan[UNIT_POS_X as usize - 1],
        I::identity::<T>(),
    );
    let vector_prefix = I::combine::<T>(cube_prefix, unit_prefix);
    #[unroll]
    for v in 0..input.vector_size() {
        vector.insert(v, I::combine::<T>(vector_prefix, vector.extract(v)));
    }

    // Vectorized store to global memory (continued)
    output[cube_id * blueprint.cube_size + UNIT_POS_X as usize] = vector;
}
