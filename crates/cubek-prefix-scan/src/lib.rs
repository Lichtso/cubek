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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) enum CubeScanStrategy {
    // 1973: https://dl.acm.org/doi/10.1109/TC.1973.5009159
    // 1986: https://dl.acm.org/doi/10.1145/7902.7903
    KoggeStoneOrHillisSteele,
    // 1990: https://www.cs.cmu.edu/~scandal/papers/CMU-CS-90-190.html
    Blelloch,
    // 1982: https://dl.acm.org/doi/10.1109/TC.1982.1675982
    BrentKung,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) enum GlobalScanStrategy {
    // 2013: https://dl.acm.org/doi/10.1145/2517327.2442539
    StreamScan,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Blueprint {
    pub cube_size: usize,
    pub cube_size_log: usize,
    pub plane_size: usize,
    pub plane_size_log: usize,
    pub plane_sized_layer_count: usize,
    pub use_task_allocator: bool,
    pub cube_scan_strategy: CubeScanStrategy,
    pub global_scan_strategy: GlobalScanStrategy,
}

pub fn inclusive_prefix_scan_launch<I: ScanInstruction>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
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

    let vector_size = tensor_vector_size_parallel(
        client
            .properties()
            .vector_sizes_in_registers(dtype.size(), 1),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let cube_size = client.properties().hardware.max_units_per_cube as usize;
    let cube_count = (input.shape[0] + cube_size * vector_size - 1) / (cube_size * vector_size); // calculate_cube_count_elemwise
    let plane_sized_layer_count = if plane_size == 1 {
        0
    } else {
        ((cube_size.ilog2() + plane_size.ilog2() - 1) / plane_size.ilog2()) as usize
    };
    let task_allocator = TensorHandle::zeros(&client, &[1], ElemType::UInt(UIntKind::U32));
    let task_status = TensorHandle::zeros(&client, &[cube_count], ElemType::UInt(UIntKind::U32));

    unsafe {
        kernels::single_pass_inclusive_prefix_scan::launch_unchecked::<I>(
            client,
            CubeCount::Static(cube_count as u32, 1, 1),
            CubeDim::new_3d(cube_size as u32, 1, 1),
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output.into_tensor_arg(),
            task_allocator.into_arg(),
            task_status.into_arg(),
            Blueprint {
                cube_size,
                cube_size_log: cube_size.ilog2() as usize,
                plane_size,
                plane_size_log: plane_size.ilog2() as usize,
                plane_sized_layer_count,
                use_task_allocator: false,
                cube_scan_strategy: CubeScanStrategy::BrentKung,
                global_scan_strategy: GlobalScanStrategy::StreamScan,
            },
            dtype,
        );
    }
    Ok(())
}
