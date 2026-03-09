pub mod aberration;
pub mod gaussian_beam;
pub mod illumination;
pub mod lens_design;
pub mod paraxial;
pub mod tracer;

pub use aberration::{
    rms_wavefront_error, strehl_marechal, strehl_marechal_waves, wavefront_map, zernike_decompose,
    zernike_defocus, zernike_piston, zernike_spherical, zernike_tilt_x, SeidelAberrations,
    SeidelCoeffs, SurfaceAberration, ZernikePolynomial, ZernikeTerm,
};
pub use gaussian_beam::{focus_gaussian, AbcdMatrix, GaussianBeam};
pub use lens_design::{AchromaticDoublet, CookeTriplet, GlassMaterial, LensMeritFunction, Singlet};
pub use paraxial::{
    CardinalPoints, ChromaticAnalysis, ParaxialImager, PupilAnalysis, SystemMatrix,
};
pub use tracer::{
    abbe_resolution, depth_of_field, f_number, numerical_aperture, OpticalSystem, Ray, Surface,
};
