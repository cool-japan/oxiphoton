pub mod brendel_bormann;
pub mod cauchy;
pub mod critical_point;
pub mod drude;
pub mod drude_lorentz;
pub mod extended_materials;
pub mod sellmeier;
pub mod tabulated;

pub use brendel_bormann::{BbOscillator, BrendelBormannModel};
pub use critical_point::{CriticalPoint, CriticalPointModel};
pub use extended_materials::{
    Diamond, InGaAs, InTinOxide, LinboRay, LithiumNiobate, SiGe, TitaniumNitride,
};
