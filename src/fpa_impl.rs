//! # Licensing
//! This Source Code is subject to the terms of the Mozilla Public License
//! version 2.0 (the "License"). You can obtain a copy of the License at
//! http://mozilla.org/MPL/2.0/.

use std::ops::Sub;

use num_traits::identities::One;
use typenum::{Cmp, Greater, Less, U0, U8, U16, U32, Unsigned};
use fpa::*;

use NearlyEq;

macro_rules! impl_fpa {
    ($bits:ident, $limit:ident) => {
        impl<FRAC> NearlyEq<Q<$bits, FRAC>, Q<$bits, FRAC>> for Q<$bits, FRAC>
        where
            FRAC: Cmp<U0, Output = Greater> + Cmp<$limit, Output = Less> + Unsigned,
            Self: PartialOrd + Clone + Sub<Self, Output=Self>,
        {
            fn eps() -> Self {
                Self::from_bits($bits::one())
            }

            fn eq(&self, other: &Self, eps: &Self) -> bool {
                let diff = if *self > *other {
                    self.clone() - other.clone()
                } else {
                    other.clone() - self.clone()
                };
                if *self == *other {
                    true
                } else {
                    diff <= *eps
                }
            }
        }
    }
}

impl_fpa!(i8, U8);
impl_fpa!(i16, U16);
impl_fpa!(i32, U32);
