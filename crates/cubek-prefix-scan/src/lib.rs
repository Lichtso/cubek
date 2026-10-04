use crate::{
    eval::strategy::{GlobalScanStrategy, ScanStrategy},
    instructions::{Maximum, Minimum, Product, ScanInstructionKind, Sum},
};
use cubecl::{
    CubeDim, client::Client, ir::UIntKind, prelude::*, std::tensor::TensorHandle,
    tensor_vector_size_parallel,
};
use error::PrefixScanError;
use instructions::ScanInstruction;

pub mod error;
pub mod eval;
pub mod instructions;
pub mod kernels;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ScanKind {
    /// Combines all preceeding elements, but not the element itself
    Exclusive,
    /// Combines all preceeding elements and the element itself
    Inclusive,
}

impl ScanKind {
    #[cfg(feature = "benchmarks")]
    pub const ALL: &[ScanKind] = &[ScanKind::Exclusive, ScanKind::Inclusive];

    pub fn label(self) -> &'static str {
        match self {
            ScanKind::Exclusive => "Exclusive",
            ScanKind::Inclusive => "Inclusive",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Blueprint {
    pub units_per_cube: usize,
    pub cube_scan_size: usize,
    pub cube_scan_size_log: usize,
    pub plane_size: usize,
    pub plane_size_log: usize,
    pub plane_sized_layer_count: usize,
    pub plane_scan_before_cube_scan: bool,
    pub use_task_allocator: bool,
    pub scan_kind: ScanKind,
    pub strategy: ScanStrategy,
}

pub fn prefix_scan_launch_generic_instruction<I: ScanInstruction>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: ScanKind,
    strategy: ScanStrategy,
) -> Result<(), PrefixScanError> {
    if input.shape.rank() != 1 {
        return Err(PrefixScanError::UnsupportedRank {
            rank: input.shape.rank(),
        });
    }

    if input.shape != output.shape {
        return Err(PrefixScanError::InputOutputShapeMismatch {
            input: input.shape.to_vec(),
            output: output.shape.to_vec(),
        });
    }

    if client.features().device_memory_scope {
        return Err(PrefixScanError::MissingDeviceMemoryScope);
    }

    let use_task_allocator = false;
    let plane_scan_before_cube_scan = false;
    let vector_size = tensor_vector_size_parallel(
        client
            .properties()
            .vector_sizes_in_registers(dtype.size(), 1),
        &input.shape,
        &input.strides,
        0,
    );
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let units_per_cube = if client.properties().hardware.num_cpu_cores.is_some() {
        // Synchronizing CPU cores inside a cube is very inefficient.
        // Better to assign them different cubes instead.
        1
    } else {
        client.properties().hardware.max_units_per_cube as usize
    };
    let mut cube_scan_size = units_per_cube;
    if plane_scan_before_cube_scan {
        cube_scan_size /= plane_size;
    }
    let cube_count = input.shape[0].div_ceil(units_per_cube * vector_size); // calculate_cube_count_elemwise
    let plane_sized_layer_count = if plane_size == 1 {
        0
    } else {
        cube_scan_size.ilog2().div_ceil(plane_size.ilog2()) as usize
    };
    let reductions = match strategy.global {
        GlobalScanStrategy::StreamScan => None,
        GlobalScanStrategy::DecoupledLookback => {
            Some(TensorHandle::zeros(client, [cube_count], dtype))
        }
    };
    let task_allocator = TensorHandle::zeros(client, [1], ElemType::UInt(UIntKind::U32));
    let task_status = TensorHandle::zeros(client, [cube_count], ElemType::UInt(UIntKind::U32));

    unsafe {
        kernels::single_pass_prefix_scan::launch_unchecked::<I>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_3d(units_per_cube as u32, 1, 1),
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            reductions
                .map(|tensor_handle| tensor_handle.into_arg())
                .unwrap_or_else(|| output.clone().into_tensor_arg()),
            task_allocator.into_arg(),
            task_status.into_arg(),
            Blueprint {
                units_per_cube,
                cube_scan_size,
                cube_scan_size_log: cube_scan_size.ilog2() as usize,
                plane_size,
                plane_size_log: plane_size.ilog2() as usize,
                plane_sized_layer_count,
                use_task_allocator,
                plane_scan_before_cube_scan,
                scan_kind,
                strategy,
            },
            dtype,
        );
    }
    Ok(())
}

pub fn prefix_scan_launch(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: ScanKind,
    scan_instruction_kind: ScanInstructionKind,
    strategy: ScanStrategy,
) -> Result<(), PrefixScanError> {
    match scan_instruction_kind {
        ScanInstructionKind::Sum => prefix_scan_launch_generic_instruction::<Sum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Product => prefix_scan_launch_generic_instruction::<Product>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Minimum => prefix_scan_launch_generic_instruction::<Minimum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Maximum => prefix_scan_launch_generic_instruction::<Maximum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
    }
}
