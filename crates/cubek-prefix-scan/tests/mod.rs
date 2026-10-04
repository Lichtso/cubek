use cubecl::{
    client::Client,
    ir::{ElemType, FloatKind},
};
use cubek_prefix_scan::{
    eval::cpu_reference::run_inclusive_prefix_scan,
    inclusive_prefix_scan_launch,
    instructions::{ScanInstruction, Sum},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, TestInput, TestOutcome, assert_equals_approx,
    launch_and_capture_outcome,
};

pub fn run_inclusive_prefix_scan_test<I: ScanInstruction>(client: &Client, input_shape: &[usize]) {
    let dtype = ElemType::Float(FloatKind::F32);
    let (input, input_data) = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .custom(vec![1f32; input_shape.iter().product()])
        .generate_with_host_data(dtype.into());
    let expected = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .custom(run_inclusive_prefix_scan::<I>(&input_data))
        .generate_without_host_data();
    let expected = HostData::from_tensor_handle(client, expected, dtype.into());
    let output = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let outcome = launch_and_capture_outcome(client, &[&output.handle], |client| {
        inclusive_prefix_scan_launch::<I>(client, input.binding(), output.clone().binding(), dtype)
            .into()
    });
    match outcome {
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
        ExecutionOutcome::Executed => {
            let actual = HostData::from_tensor_handle(client, output, dtype.into());
            assert_equals_approx(&actual, &expected, 0.0f32)
                .as_test_outcome()
                .enforce();
        }
    }
}

#[test]
fn test_inclusive_prefix_scan_sum() {
    let client = cubecl::test_device().client();
    run_inclusive_prefix_scan_test::<Sum>(&client, &[1000000]);
}
