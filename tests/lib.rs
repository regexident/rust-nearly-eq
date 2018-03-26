#![cfg_attr(feature = "i128", feature(i128_type))]

#[cfg(feature = "num-complex")]
extern crate num_complex;

#[cfg(feature = "num-rational")]
extern crate num_rational;

#[cfg(feature = "num-traits")]
extern crate num_traits;

#[cfg(feature = "ndarray")]
extern crate ndarray;

#[cfg(feature = "fpa")]
extern crate fpa;

#[macro_use]
extern crate nearly_eq;

#[cfg(feature = "num-complex")]
use num_complex::Complex;

#[cfg(feature = "num-rational")]
use num_rational::Rational;

#[cfg(feature = "ndarray")]
use ndarray::{ArrayD, IxDyn, arr1, arr2, arr3};

#[cfg(feature = "fpa")]
use fpa::*;

use std::rc::Rc;

use std::sync::Arc;

use std::cell::{Cell, RefCell};

#[cfg(test)]
mod assert {
    mod f32 {
        #[test]
        fn success() {
            assert_nearly_eq!(8.0_f32, 8.0_f32 + 1e-7);
        }

        #[test]
        fn success_eps() {
            assert_nearly_eq!(3.0_f32, 4.0_f32, 2.0_f32);
        }

        #[test]
        #[should_panic]
        fn failure() {
            assert_nearly_eq!(8.0_f32, 8.0_f32 - 1e-5);
        }
    }

    mod f64 {
        #[test]
        fn success() {
            assert_nearly_eq!(8.0_f64, 8.0_f64 + 1e-15);
        }

        #[test]
        fn success_eps() {
            assert_nearly_eq!(3.0_f64, 4.0_f64, 2.0_f64);
        }

        #[test]
        #[should_panic]
        fn failure() {
            assert_nearly_eq!(8.0_f64, 8.0_f64 - 1e-5);
        }
    }
}

#[cfg(test)]
mod debug_assert {
    mod f32 {
        #[test]
        #[cfg(debug_assertions)]
        fn success() {
            debug_assert_nearly_eq!(8.0_f32, 8.0_f32 + 1e-7);
        }

        #[test]
        #[cfg(debug_assertions)]
        fn success_eps() {
            debug_assert_nearly_eq!(3.0_f32, 4.0_f32, 2.0_f32);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic]
        fn failure() {
            debug_assert_nearly_eq!(8.0_f32, 8.0_f32 - 1e-5);
        }
    }

    mod f64 {
        #[test]
        #[cfg(debug_assertions)]
        fn success() {
            debug_assert_nearly_eq!(0.0_f64, 1e-12 as f64);
        }

        #[test]
        #[cfg(debug_assertions)]
        fn success_eps() {
            debug_assert_nearly_eq!(3.0_f64, 4.0_f64, 2.0_f64);
        }

        #[test]
        #[cfg(debug_assertions)]
        #[should_panic]
        fn failure() {
            debug_assert_nearly_eq!(8.0_f64, 8.0_f64 - 1e-5);
        }
    }
}

#[cfg(test)]
mod complex {
    use super::*;

    mod vector {
        #[test]
        fn success() {
            let left = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let right = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.01];
            let right = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure_len() {
            let left = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let right = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(left, right);
        }
    }

    mod slice {
        #[test]
        fn success() {
            let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let right = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(&left as &[f32], &right as &[f32]);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.01];
            let right = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(&left as &[f32], &right as &[f32]);
        }

        #[test]
        #[should_panic]
        fn failure_len() {
            let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let right = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(&left as &[f32], &right as &[f32]);
        }
    }

    mod array {
        #[test]
        fn success() {
            let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let right = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.01];
            let right = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            assert_nearly_eq!(left, right);
        }
    }

    mod option {

        mod both_none {
            #[test]
            fn success() {
                let left: Option<f64> = Option::None;
                let right = Option::None;
                assert_nearly_eq!(left, right);
            }
        }

        mod both_some {
            #[test]
            fn success() {
                let left = Option::Some(0.0_f64);
                let right = Option::Some(1e-12);
                assert_nearly_eq!(left, right);
            }

            #[test]
            #[should_panic]
            fn failure() {
                let left = Option::Some(0.0_f64);
                let right = Option::Some(1.0_f64);
                assert_nearly_eq!(left, right);
            }
        }

        mod left_some {
            #[test]
            #[should_panic]
            fn failure() {
                let left = Option::Some(0.0_f64);
                let right = Option::None;
                assert_nearly_eq!(left, right);
            }
        }

        mod right_some {
            #[test]
            #[should_panic]
            fn failure() {
                let left: Option<f64> = Option::None;
                let right = Option::Some(0.0_f64);
                assert_nearly_eq!(left, right);
            }
        }
    }

    mod rc {
        use super::*;

        #[test]
        fn success() {
            let left = Rc::new(1.0);
            let right = Rc::new(1.0);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Rc::new(1.0);
            let right = Rc::new(1.00001);
            assert_nearly_eq!(left, right);
        }
    }

    mod arc {
        use super::*;

        #[test]
        fn success() {
            let left = Arc::new(1.0);
            let right = Arc::new(1.0);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Arc::new(1.0);
            let right = Arc::new(1.00001);
            assert_nearly_eq!(left, right);
        }
    }

    mod weak {
        use super::*;

        #[test]
        fn success() {
            let left = Rc::new(1.0);
            let right = Rc::new(1.0);
            assert_nearly_eq!(Rc::downgrade(&left), Rc::downgrade(&right));
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Rc::new(1.0);
            let right = Rc::new(1.00001);
            assert_nearly_eq!(Rc::downgrade(&left), Rc::downgrade(&right));
        }
    }

    mod cell {
        use super::*;

        #[test]
        fn success() {
            let left = Cell::new(1.0);
            let right = Cell::new(1.0);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Cell::new(1.0);
            let right = Cell::new(1.00001);
            assert_nearly_eq!(left, right);
        }
    }

    mod refcell {
        use super::*;

        #[test]
        fn success() {
            let left = RefCell::new(1.0);
            let right = RefCell::new(1.0);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = RefCell::new(1.0);
            let right = RefCell::new(1.00001);
            assert_nearly_eq!(left, right);
        }
    }
}

#[cfg(test)]
mod primitive {
    macro_rules! type_impls {
        ($($T:ident)+) => {
            $(
                mod $T {
                    #[test]
                    fn success() {
                        assert_nearly_eq!(0 as $T, 0 as $T);
                    }

                    #[test]
                    #[should_panic]
                    fn failure() {
                        assert_nearly_eq!(0 as $T, 1 as $T);
                    }
                }
            )+
        }
    }

    type_impls! { i8 i16 i32 i64 u8 u16 u32 u64 }

    #[cfg(feature = "i128")]
    type_impls! { i128 u128 }
}

#[cfg(test)]
mod third_party {
    use super::*;

    #[cfg(feature = "num-rational")]
    mod num_rational {
        use super::*;

        #[test]
        fn success_eps() {
            let left = Rational::new(1, 1000);
            let right = Rational::new(1, 1001);
            let eps = Rational::new(1, 10000);
            assert_nearly_eq!(left, right, eps);
        }

        #[test]
        fn success() {
            let left = Rational::new(1, 1000);
            let right = Rational::new(1, 1000);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Rational::new(1, 1000);
            let right = Rational::new(1, 1001);
            let eps = Rational::new(1, 1000000000);
            assert_nearly_eq!(left, right, eps);
        }
    }

    #[cfg(feature = "num-complex")]
    mod num_complex {
        use super::*;

        #[test]
        fn success() {
            let left = Complex::new(1.0_f64, 0.0);
            let right = Complex::new(1.0_f64, 1e-12);
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = Complex::new(1.0_f64, 0.0);
            let right = Complex::new(1.0_f64, 1e-8);
            assert_nearly_eq!(left, right);
        }
    }

    #[cfg(feature = "ndarray")]
    mod ndarray {
        use super::*;

        mod ndarray1d {
            use super::*;

            #[test]
            fn success() {
                let left = arr1(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                let right = arr1(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                assert_nearly_eq!(left, right);
            }

            #[test]
            #[should_panic]
            fn failure() {
                let left = arr1(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0001]);
                let right = arr1(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
                assert_nearly_eq!(left, right);
            }
        }

        mod ndarray2d {
            use super::*;

            #[test]
            fn success() {
                let left = arr2(&[[1.0_f64, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);
                let right = arr2(&[[1.0_f64, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);
                assert_nearly_eq!(left, right);
            }

            #[test]
            #[should_panic]
            fn failure_val() {
                let left = arr2(&[[1.0_f64, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0001]]);
                let right = arr2(&[[1.0_f64, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);
                assert_nearly_eq!(left, right);
            }

            #[test]
            #[should_panic]
            fn failure_len() {
                let left = arr2(&[[1.0_f64, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]);
                let right = arr2(&[
                    [1.0_f64, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                    [7.0, 8.0],
                    [9.0, 10.0],
                ]);
                assert_nearly_eq!(left, right);
            }
        }

        mod ndarray3d {
            use super::*;

            #[test]
            fn success() {
                let left = arr3(&[[[1.0_f64, 2.0], [4.0, 5.0]], [[6.0, 7.0], [9.0, 10.0]]]);
                let right = arr3(&[[[1.0_f64, 2.0], [4.0, 5.0]], [[6.0, 7.0], [9.0, 10.0]]]);
                assert_nearly_eq!(left, right);
            }

            #[test]
            #[should_panic]
            fn failure() {
                let left = arr3(&[[[1.0_f64, 2.0], [4.0, 5.0]], [[6.0, 7.0], [9.0, 10.0001]]]);
                let right = arr3(&[[[1.0_f64, 2.0], [4.0, 5.0]], [[6.0, 7.0], [9.0, 10.0]]]);
                assert_nearly_eq!(left, right);
            }
        }

        mod ndarraynd {
            use super::*;

            #[test]
            #[should_panic]
            fn failure() {
                let left = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4, 5, 6, 7]));
                let right = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4, 5, 6]));
                assert_nearly_eq!(left, right);
            }
        }
    }

    #[cfg(feature = "fpa")]
    mod fixed {
        use super::*;

        #[test]
        fn success_eps() {
            let left = I16F16(42.000_f32).unwrap();
            let right = I16F16(42.001_f32).unwrap();
            let eps = I16F16(00.001_f32).unwrap();
            assert_nearly_eq!(left, right, eps);
        }

        #[test]
        fn success() {
            let left = I16F16(42.00000_f32).unwrap();
            let right = I16F16(42.00001_f32).unwrap();
            assert_nearly_eq!(left, right);
        }

        #[test]
        #[should_panic]
        fn failure() {
            let left = I16F16(42.000_f32).unwrap();
            let right = I16F16(42.001_f32).unwrap();
            assert_nearly_eq!(left, right);
        }
    }
}
