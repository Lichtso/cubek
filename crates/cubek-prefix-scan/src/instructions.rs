use cubecl::prelude::*;

#[cube]
pub trait ScanInstruction: Send + Sync + 'static + std::fmt::Debug + CubeType + Sized {
    fn identity<T: Numeric>() -> T;
    fn combine<T: Numeric>(a: T, b: T) -> T;
}

#[derive(CubeType, Debug)]
pub struct Sum {}
#[cube]
impl ScanInstruction for Sum {
    fn identity<T: Numeric>() -> T {
        T::from_int(0)
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        a + b
    }
}

#[derive(CubeType, Debug)]
pub struct Product {}
#[cube]
impl ScanInstruction for Product {
    fn identity<T: Numeric>() -> T {
        T::from_int(1)
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        a * b
    }
}

#[derive(CubeType, Debug)]
pub struct Minimum {}
#[cube]
impl ScanInstruction for Minimum {
    fn identity<T: Numeric>() -> T {
        T::max_value()
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        select(a < b, a, b)
    }
}

#[derive(CubeType, Debug)]
pub struct Maximum {}
#[cube]
impl ScanInstruction for Maximum {
    fn identity<T: Numeric>() -> T {
        T::min_value()
    }
    fn combine<T: Numeric>(a: T, b: T) -> T {
        select(a > b, a, b)
    }
}
