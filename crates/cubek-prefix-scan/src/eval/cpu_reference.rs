use crate::instructions::ScanInstruction;
use cubek_test_utils::HostData;

pub fn run_inclusive_prefix_scan<I: ScanInstruction>(input: &HostData) -> Vec<f32> {
    (0..input.shape[0])
        .scan(0.0f32, |accumulator, index| {
            *accumulator = I::combine::<f32>(*accumulator, input.get_f32(&[index]));
            Some(*accumulator)
        })
        .collect::<Vec<_>>()
}
