use cubecl::{
    client::Client,
    ir::{ElemType, FloatKind},
};
use cubek_prefix_scan::{
    ScanKind,
    eval::{cpu_reference::cpu_reference_prefix_scan, strategy::strategies},
    instructions::ScanInstructionKind,
    prefix_scan_launch,
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, TestInput, TestOutcome, assert_equals_approx,
    launch_and_capture_outcome,
};

pub fn run_prefix_scan_test(
    client: &Client,
    input_shape: &[usize],
    scan_kind: ScanKind,
    scan_instruction_kind: ScanInstructionKind,
) {
    let dtype = ElemType::Float(FloatKind::F32);
    let (input, input_data) = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .custom(vec![1f32; input_shape.iter().product()])
        .generate_with_host_data(dtype.into());
    let expected = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .custom(cpu_reference_prefix_scan(
            &input_data,
            scan_kind,
            scan_instruction_kind,
        ))
        .generate_without_host_data();
    let expected = HostData::from_tensor_handle(client, expected, dtype.into());
    let output = TestInput::builder(client.clone(), input_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    for strategy in strategies() {
        let outcome = launch_and_capture_outcome(client, &[&output.handle], |client| {
            prefix_scan_launch(
                client,
                input.clone().binding(),
                output.clone().binding(),
                dtype,
                scan_kind,
                scan_instruction_kind,
                strategy.value,
            )
            .into()
        });
        match outcome {
            ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e).enforce(),
            ExecutionOutcome::Executed => {
                let actual = HostData::from_tensor_handle(client, output.clone(), dtype.into());
                assert_equals_approx(&actual, &expected, 0.0f32)
                    .as_test_outcome()
                    .enforce();
            }
        }
    }
}

const TEST_DIM: &[usize] = &[1000000];

#[test]
fn test_exclusive_prefix_sum() {
    let client = cubecl::test_device().client();
    run_prefix_scan_test(
        &client,
        TEST_DIM,
        ScanKind::Exclusive,
        ScanInstructionKind::Sum,
    );
}

#[test]
fn test_inclusive_prefix_sum() {
    let client = cubecl::test_device().client();
    run_prefix_scan_test(
        &client,
        TEST_DIM,
        ScanKind::Inclusive,
        ScanInstructionKind::Sum,
    );
}

#[test]
fn test_exclusive_prefix_product() {
    let client = cubecl::test_device().client();
    run_prefix_scan_test(
        &client,
        TEST_DIM,
        ScanKind::Exclusive,
        ScanInstructionKind::Product,
    );
}

#[test]
fn test_inclusive_prefix_product() {
    let client = cubecl::test_device().client();
    run_prefix_scan_test(
        &client,
        TEST_DIM,
        ScanKind::Inclusive,
        ScanInstructionKind::Product,
    );
}
