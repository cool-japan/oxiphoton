pub use crate::error::{OxiPhotonError, Result};

// Units
pub use crate::units::electromagnetic::{Frequency, WaveNumber, Wavelength};
pub use crate::units::field::{ElectricField, Intensity, MagneticField, Poynting};
pub use crate::units::geometric::{Angle, FocalLength, NumericalAperture};
pub use crate::units::optical::{Permeability, Permittivity, RefractiveIndex};

// Physical constants
pub use crate::units::conversion::{EPSILON_0, MU_0, SPEED_OF_LIGHT, Z0};

// Materials
pub use crate::material::{Cauchy, Drude, DrudeLorentz, Sellmeier, Tabulated};
pub use crate::material::{DispersiveMaterial, MaterialDatabase};

// S-Matrix / Transfer Matrix
#[cfg(feature = "smatrix")]
pub use crate::smatrix::transfer_matrix::{
    ConstantMaterial, Layer, TransferMatrix, TransferMatrixResult,
};
#[cfg(feature = "smatrix")]
pub use crate::smatrix::Polarization;

// Geometry
pub use crate::geometry::primitives::{Circle2d, Rect2d, Shape2d};

// FDTD — core 1D/2D
#[cfg(feature = "fdtd")]
pub use crate::fdtd::{
    BoundaryConfig, DftBox2d, DftMonitor1d, Dimensions, Fdtd1d, Fdtd2dTe, FluxMonitor1d,
    FluxMonitorDft, GaussianEnvelope, GaussianModulated, GridSpacing, PlaneWaveSource,
    SourceWaveform, TfsfSource,
};

// FDTD — 3D engine types
#[cfg(feature = "fdtd-3d")]
pub use crate::fdtd::{
    // 3D dispersive engines (Drude + Lorentz ADE)
    AdeCoeffs3d,
    Axis3d,
    Checkpoint3d,
    CwWaveform3d,
    DftProbe3d,
    DrudeParams,
    // 3D base solver + field/source types
    Fdtd3d,
    Fdtd3dDrude,
    Fdtd3dLorentz,
    Fdtd3dMaterial,
    FieldComponent3d,
    FieldProbe3d,
    GaussianPulse3d,
    GaussianWaveform3d,
    // 3D nonlinear engines
    KerrFdtd3d,
    KerrMedium,
    LorentzParams,
    PlaneMonitor3d,
    RamanFdtd3d,
    Shg3d,
    SourceType3d,
    SourceWaveform3d,
    // 3D Yee grid
    Yee3d,
};

// Mode solvers
pub use crate::mode::{
    DirectionalCouplerCmt, FdMode, FdModeSolver1d, FemMode1d, FemModeSolver1d, GratingCoupler,
    NanocavityCmt, ResonatorCmt, SlabMode, SlabWaveguide, TaperedCoupler, TemporalCmt,
};

// BPM
pub use crate::bpm::{
    BiDirectionalBpm1d, BidirectionalBpm, BidirectionalBpmSection, FdBpm1d, FftBpm1d, JonesMatrix,
    JonesVector, VectorBpm1d,
};

// Ray optics / paraxial
pub use crate::ray::{
    rms_wavefront_error, strehl_marechal_waves, wavefront_map, zernike_decompose, zernike_defocus,
    zernike_piston, zernike_spherical, zernike_tilt_x, AchromaticDoublet, CardinalPoints,
    ChromaticAnalysis, CookeTriplet, GlassMaterial, ParaxialImager, PupilAnalysis, SeidelCoeffs,
    Singlet, SurfaceAberration, SystemMatrix,
};

// EME / eigenmode propagation
#[cfg(feature = "smatrix")]
pub use crate::smatrix::{
    confinement_loss, coupling_efficiency, effective_loss_db_per_cm, mode_loss_db_per_cm,
    overlap_integral, overlap_matrix, propagation_loss_db, EigenMode, EigenmodePropagator,
};

// Waveguide devices (all)
#[cfg(feature = "siph-devices")]
pub use crate::devices::waveguide::{
    AdiabaticTaper, MmiSplitter, MultimodeWaveguide, RidgeWaveguide, SlotWaveguide, StripWaveguide,
    TaperProfile, WaveguideBend,
};

// Couplers (expanded)
#[cfg(feature = "siph-devices")]
pub use crate::devices::coupler::{
    ApodizedGratingCoupler, AsymmetricCoupler, DirectionalCoupler, EfficiencyVsTilt,
    GratingArray2d, Mmi1x2, MmiCoupler,
};

// Resonators (expanded)
#[cfg(feature = "siph-devices")]
pub use crate::devices::resonator::{
    slow_light_bandwidth_hz, CoupledL3Resonators, CoupledResonatorOW, CoupledRingFilter,
    FabryPerot, L3CavityEstimate, RingResonator, W1WaveguideDispersion,
};

// Metalens layout (expanded)
pub use crate::devices::metalens::{AchromaticMetalens, FillFactorMap, ZonePlate};

// EO modulators + plasma dispersion
#[cfg(feature = "siph-devices")]
pub use crate::devices::modulator::plasma_dispersion::{
    PinDiodeModel, SiPlasmaDispersion as SiPlasmaDispersionNew,
};
#[cfg(feature = "siph-devices")]
pub use crate::devices::modulator::{
    EoCrystal, EoModulatorBandwidth, LongitudinalPockelsCell, MziModulator, PockelsModulator,
    SiPlasmaDispersion, TransversePockelsCell,
};

// Detectors
#[cfg(feature = "siph-devices")]
pub use crate::devices::detector::{
    AvalanchePhotodetector, DetectorBandwidth, DetectorNoise, Photodiode, SpectralResponsivity,
};

// Extended dispersive materials
pub use crate::material::dispersive::brendel_bormann::BrendelBormannModel;
pub use crate::material::dispersive::critical_point::CriticalPointModel;
pub use crate::material::dispersive::extended_materials::{
    Diamond, InGaAs, InTinOxide, LithiumNiobate, SiGe, TitaniumNitride,
};

// Adaptive Optics
pub use crate::adaptive_optics::atmosphere::{
    AtmosphericTurbulence, LayeredAtmosphere, PhaseScreen, TurbulentLayer,
};
pub use crate::adaptive_optics::control::{
    ClosedLoopMetrics, IntegralController, ModalController, PredictiveController,
};
pub use crate::adaptive_optics::deformable_mirror::{
    zernike_ansi, DeformableMirror, SegmentedMirror, ZernikeCorrector,
};
pub use crate::adaptive_optics::wavefront_sensor::{
    CurvatureSensor, PyramidSensor, ShackHartmannSensor,
};

// I/O
pub use crate::io::{
    KnowledgeGraph, LumericalDomain, LumericalParser, LumericalSimulation, PhotonicSimExporter,
};

// Coherence optics
pub use crate::coherence::mutual_coherence::{
    degree_of_coherence, van_cittert_zernike_theorem, CoherenceError, CoherenceMatrix,
    CrossSpectralDensity, MutualCoherenceFunction, SchellModelBeam,
};
pub use crate::coherence::spatial::{
    LateralCoherenceFunction, PartiallyCoherentBeam, PropagatingCoherence, SpatialCoherence,
    SpatialCoherenceError,
};
pub use crate::coherence::speckle::{
    ObjectiveSpeckleSize, SpeckleCorrelation, SpeckleError, SpeckleReduction, SpeckleStatistics,
};
pub use crate::coherence::temporal::{
    MichelsonVisibility, PowerSpectralDensity, SpectralShape, TemporalCoherence,
    TemporalCoherenceError,
};

// Photonic crystal slab, extended topology, nonlinear PhC
#[cfg(feature = "photonic-crystal")]
pub use crate::photonic_crystal::{
    // Extended topology
    BerryPhase,
    // Slab structures
    CavityMode,
    CavityPolarization,
    ChernNumber,
    DefectType,
    EdgeDirection,
    HoleShape,
    // Nonlinear PhC
    PhCNonlinearEnhancement,
    PhCSlabStructure,
    PhCW1Waveguide,
    PointDefectCavity,
    SlabLattice,
    SlowLightShg,
    SshPhotonicChain,
    TopologicalEdgeState,
    ValleyPhotonicCrystal,
};

// Fiber optics — solitons and supercontinuum
#[cfg(feature = "fiber")]
pub use crate::fiber::{
    FundamentalSoliton, GnlseSolver, HigherOrderSoliton, OpticalWaveBreaking, PeregineSoliton,
    PumpingRegime, ScFiberType, SolitonTrap, SupercontinuumSource,
};

// MEMS & Microresonator Physics
pub use crate::mems::cantilever::{CantileverMaterial, OpticalCantilever};
pub use crate::mems::coupling::{DiskResonator, WgmMicroresonator};
pub use crate::mems::fabry_perot_mems::{FpMemsError, MemosFabryPerot};
pub use crate::mems::mems_mirror::{GimbalMirror, MemosTiltMirror, MemosVoa};

// Microwave Photonics
pub use crate::microwave_photonics::adc::{PhotonicAdc, PhotonicChannelizer};
pub use crate::microwave_photonics::beamforming::{
    BfnArchitecture, OpticalBfn, PhotonicBeamformer,
};
pub use crate::microwave_photonics::link::{
    AnalogPhotonicLink, EoModulatorType, LinkBudget, MzmBias, PhotodetectorParams,
};
pub use crate::microwave_photonics::rf_filter::{
    PhotonicHilbertTransformer, PhotonicRfFilter, RingResonatorRfFilter,
};

// Optical Computing & Photonic Neural Networks
pub use crate::optical_computing::mzi_mesh::{ClementsArch, MziCell, ReckArch};
pub use crate::optical_computing::optical_matrix::{
    OpticalOuterProduct, OpticalSystolicArray, WdmMac,
};
pub use crate::optical_computing::photonic_nn::{
    ActivationFn, D2nnLayer, PhotonicLayer, PhotonicNeuralNetwork,
};
pub use crate::optical_computing::reservoir::{EchoStateNetwork, OpticalReservoir};

// Quantum Photonic Computing
pub use crate::quantum_photonics::boson_sampling::{
    BosonSamplingCertificate, GaussianBosonSampler,
};
pub use crate::quantum_photonics::fock_state::{
    BellState, FockSuperposition, MultiModeFockState, PnrdMeasurement,
};
pub use crate::quantum_photonics::hom_effect::{
    HomInterferometer, IndistinguishabilityMeasurement, MultiPhotonHom,
};
pub use crate::quantum_photonics::linear_optical::{
    hafnian, permanent, KlmCnot, LinearOpticalNetwork, LopGate, MziMesh,
};

// X-ray & EUV Optics
pub use crate::xray::{
    // Utility conversions
    kev_to_wavelength_m,
    wavelength_m_to_kev,
    BraggCrystal,
    CompoundRefractiveLens,
    // Bragg diffraction
    CrystalMaterial,
    EuvMirror,
    FreeElectronLaser,
    // Fresnel zone plate and focusing optics
    FresnelZonePlate,
    JohannSpectrometer,
    KbMirror,
    MultilayerMirror,
    // Synchrotron radiation
    SynchrotronSource,
    Undulator,
    // Multilayer mirrors
    XrayMaterial,
};

// Photonic Sensors & LiDAR
pub use crate::photonic_sensors::chemical::{EvanescentSensor, SprMetal, SprSensor, WgmBiosensor};
pub use crate::photonic_sensors::gyroscope::{
    FiberOpticGyroscope, IntegratedGyroscope, RingLaserGyroscope, EARTH_ROTATION_RATE_RAD_S,
};
pub use crate::photonic_sensors::inertial::{
    IntegratedStrainGauge, PhotonicAccelerometer, PhotonicPressureSensor,
};
pub use crate::photonic_sensors::lidar::{FmcwLidar, LidarScanner, PhotonCountingLidar, TofLidar};

// Optical Frequency Combs & Precision Metrology
pub use crate::frequency_comb::comb::{CombState, FrequencyComb, KerrMicrocomb, C0 as COMB_C0};
pub use crate::frequency_comb::spectroscopy::{
    DirectCombSpectroscopy, DualCombSpectroscopy, HhgGas, HhgSource,
};
pub use crate::frequency_comb::stabilization::{
    AtomType, CombPll, F2fInterferometer, OpticalClock,
};
pub use crate::frequency_comb::timing::{AllanDeviation, ClockComparison, FiberFrequencyTransfer};

// Quantum Entanglement & Photon Pair Sources
pub use crate::entanglement::bell_inequality::{
    ChTest, ChshTest, LoopholeFreeAnalysis, MerminTest, CHSH_CLASSICAL_BOUND,
    CHSH_DETECTION_THRESHOLD, CHSH_TSIRELSON_BOUND,
};
pub use crate::entanglement::entanglement_measures::{
    binary_entropy, DensityMatrix, PolarizationEntanglement, TimeBinEntanglement,
};
pub use crate::entanglement::qkd::{Bb84Protocol, CvQkd, E91Protocol};
pub use crate::entanglement::spdc::{PhaseMatchingType, SpdcCrystal, SpdcSource};

// Metasurfaces & Flat Optics
pub use crate::metasurface::geometric_phase::{
    CircPolarization, MetasurfaceFunction, PbBeamSplitter, PbPhaseElement,
    SpinMultiplexedMetasurface,
};
pub use crate::metasurface::metalens::{
    Metalens, MetalensDoublet, MetalensUnitCellType, TuningMechanism, VarifocalMetalens,
};
pub use crate::metasurface::reflectarray::{HolographicMetasurface, ReflectArray, RisElement};
pub use crate::metasurface::unit_cell::{
    DielectricPillar, HuygensMetasurface, PlasmonicAntenna, PlasmonicMetal, VAntenna,
};

// Structured Light & OAM Beams
pub use crate::structured_light::bessel_beam::{
    AiryBeam, BesselBeam, BesselGaussBeam, VectorBeam, VectorBeamType,
};
pub use crate::structured_light::laguerre_gaussian::{HgBeam, LgBeam, OpticalVortex};
pub use crate::structured_light::oam_multiplexing::{
    OamModeSorter, OamMultiplexLink, SpiralPhasePlate,
};
pub use crate::structured_light::spatiotemporal::{
    FlyingFocus, PulsedOamBeam, SpaceTimeWavePacket,
};

// Photonic DSP
pub use crate::photonic_dsp::coherent_receiver::{
    CarrierPhaseEstimation, CdCompensator, FrequencyOffsetEstimator, OpticalHybrid, PolDemux,
};
pub use crate::photonic_dsp::dsp_algorithms::{
    ber_dp_qpsk_from_osnr, ber_to_q_factor, dft, erfc_approx, erfinv_approx, gram_schmidt, idft,
    osnr_to_snr, q_factor_to_ber, required_osnr, EyeDiagram, FecCode, WelchPsd, WindowType,
};
pub use crate::photonic_dsp::filters::{
    EqAlgorithm, OpticalEqualizer, OpticalFirFilter, RingResonatorFilter,
};
pub use crate::photonic_dsp::modulation_formats::{
    ConstellationPoint, Dp16Qam, DpQpsk, OfdmModulator, QamConstellation, ShapedConstellation,
};

// Optical Network Design & WDM Systems
pub use crate::optical_network::impairments::{FwmEfficiency, PmdAnalysis, SrsTilt, XpmPenalty};
pub use crate::optical_network::link_design::{FiberType, OpticalSpan, WdmLink};
pub use crate::optical_network::roadm::{
    OpticalCrossConnect, OxcGranularity, RoadmNode, WavelengthSelectiveSwitch,
};
pub use crate::optical_network::wdm_system::{
    ItuChannelPlan, ItuGrid, WdmLineSystem, WdmModFormat,
};

// PIC Design Tools
pub use crate::pic_design::optimization::{
    InverseDesignObjective, MziOptimizer, PsoOptimizer, RingOptimizer,
};
pub use crate::pic_design::pdk::{
    freq_to_wavelength, wavelength_to_freq, DcSpec, GcSpec, MmiSpec, PicComponentLibrary,
    PicProcess, RingSpec, SiNProcess, SoiProcess, YJunctionSpec,
};
pub use crate::pic_design::routing::{
    PicRouter, RouteSegment, SBend, TaperShape, WaveguideBend as PicWaveguideBend,
    WaveguideCrossing, WaveguideRoute, WaveguideTaper,
};
pub use crate::pic_design::verification::{
    CircuitVerifier, DesignRuleChecker, DrcSeverity, DrcViolation, ThermalAnalyzer,
};

// Ultrafast pulse characterisation
pub use crate::ultrafast::autocorrelation::{
    CrossCorrelation, IntensityAutocorrelation, InterferometricAutocorrelation, PulseShape,
};
pub use crate::ultrafast::frog::{
    ChirpedGaussianPulse, FrogError, FrogTrace, FrogType, Grenouille,
};
pub use crate::ultrafast::pulse_shaping::{
    ActivePulseCompressor, DazzlerShaper, FourFPulseShaper, PulseShaperError,
};
pub use crate::ultrafast::spider::{
    MiipsMeasurement, SpectralPhaseAnalysis, SpiderError, SpiderMeasurement, TaylorCoeffs,
};

// Optical Amplifiers
pub use crate::amplifiers::edfa::{Edfa, EdfaCascade, PumpDirection, WdmChannel, WdmEdfa};
pub use crate::amplifiers::noise::{
    attenuator_noise_figure_db, bandwidth_nm_to_hz, nf_to_nsp, AmplifierNoiseAnalysis,
    CascadedNoiseAnalysis, LaserLinewidth, RinAnalysis,
};
pub use crate::amplifiers::raman_amp::{
    photon_energy_j, raman_bandwidth_fwhm_cm_inv, raman_gain_profile, stokes_wavelength,
    wavelength_to_raman_shift, LumpedRamanAmplifier, RamanAmplifier, RamanFiberType,
};
pub use crate::amplifiers::soa::{linear_gain_to_db, modal_gain_to_linear, Soa, SoaXgm};

// Near-field Optics & Nanophotonics
pub use crate::nearfield::ldos::{CavityQedCoupling, Ldos, SpontaneousEmission};
pub use crate::nearfield::nanocavity::{
    BowtiAntenna, BowtieMaterial, MimNanocavity, NanoparticleOnMirror, PhCCavityType, PhCNanocavity,
};
pub use crate::nearfield::optical_force_nano::{NanoparticleForce, NearFieldForce, NtaSimulator};
pub use crate::nearfield::sers::{NanotagType, SersNanotag, SersSubstrate, TersSetup};

// Single-Photon Emitters — Quantum Dots & Solid-State Colour Centres
pub use crate::single_photon::cavity_emitter::{
    purcell_factor, CavityEnhancedSource, CavityType, ExtractionEfficiency, SinglePhotonBenchmark,
};
pub use crate::single_photon::color_center::{HbnDefect, NvCenter, NvCharge, SivCenter, SnvCenter};
pub use crate::single_photon::photon_statistics::{G2Function, HbtSetup, PhotonNumberDistribution};
pub use crate::single_photon::quantum_dot::{BiexcitonCascade, InAsQd, QdEnsemble};

// Spatial Division Multiplexing (SDM) & Few-Mode Fiber
pub use crate::sdm::few_mode_fiber::{CoreLayout, FewModeFiber, MulticoreFiber};
pub use crate::sdm::lp_modes::{bessel_j, bessel_k, LpMode, StepIndexFiberModes};
pub use crate::sdm::mdm_system::{MdmModFormat, MdmSystem, ModegroupDemux, SdmCapacityComparison};
pub use crate::sdm::mode_coupling::{
    FmfMimoEqualizer, LpgModeConverter, PhotonicLantern, RandomModeCoupling,
};

// FSO & Atmospheric Propagation
pub use crate::fso::beam_wander::{BeamWander, TipTiltCorrection};
pub use crate::fso::fso_link::{AerosolType, AtmosphericExtinction, FsoLink, FsoModulation};
pub use crate::fso::pointing::{PointingSystem, SatelliteOpticalLink};
pub use crate::fso::turbulence::{
    AtmosphericPath, Cn2Profile, GammaGammaDistribution, LogNormalScintillation, TurbulenceRegime,
};

// Nanolaser & Micro-Laser Physics
pub use crate::nanolaser::laser_noise::{
    laser_noise_transfer, LaserFrequencyNoise, PartitionNoise, RinSpectrum,
};
pub use crate::nanolaser::nanolaser_physics::{
    GainMaterial, GainMedium, PhcNanolaser, PlasmonicCavity, Spaser, Vecsel,
};
pub use crate::nanolaser::rate_equations::{GeneralizedRateEquations, SmallSignalAnalysis};
pub use crate::nanolaser::vcsel::{DbrMaterial, Vcsel, VcselArray};

// Optical Trapping & Manipulation
pub use crate::optical_trapping::brownian::{
    diffusion_coefficient, faxen_drag_correction, stokes_drag, LangevinSimulator,
};
pub use crate::optical_trapping::forces::{GaussianTrap, MieParticle, RayleighParticle};
pub use crate::optical_trapping::photophoresis::{thermophoretic_force, PhotophoreticForce};
pub use crate::optical_trapping::trap::{DualBeamTrap, OpticalPotential, TrapCharacterization};

// Beam Quality (M², ISO 11146)
pub use crate::beam_quality::{
    brightness_w_per_m2_sr, hermite_gaussian_m2, laguerre_gaussian_m2, synthetic_gaussian_caustic,
    BeamCaustic, BeamMeasurement, BeamProfile1d, BeamProfile2d, BeamQuality, Iso11146Measurement,
    Iso11146Result,
};

// Nonlinear Optical Microscopy (SHG/THG, CARS/SRS, STED, FCS)
pub use crate::nonlinear_microscopy::cars::{
    CarsSetup, CarsSignal, RamanSusceptibility, SrsDetector,
};
pub use crate::nonlinear_microscopy::fcs::{FccsMeasurement, FcsFitter, FcsSetup};
pub use crate::nonlinear_microscopy::shg_microscopy::{CollagenShg, ShgMicroscope, ThgMicroscope};
pub use crate::nonlinear_microscopy::sted::{Fluorophore as StedFluorophore, Sted3d, StedBeam};

// Topological Photonics
pub use crate::topological_photonics::chern_insulator::{
    berry_curvature_map, chern_from_curvature_map, QwzModel,
};
pub use crate::topological_photonics::ssh_chain::{PhotonicSshResonator, SshChain};
pub use crate::topological_photonics::topological_edge_states::{
    AnomalousQhpc, PhotonicTopologicalInsulator, TopologicalEdgeState as TopologicalEdgeStateNew,
};
pub use crate::topological_photonics::valley_hall::{ValleyHallPhC, ValleyKinkState};

// Temporal Photonics — Floquet, OPA, time refraction, photonic time crystals
pub use crate::temporal_photonics::floquet_theory::{FloquetCavity, ModulatedCavity};
pub use crate::temporal_photonics::parametric_amplification::{
    OpticalParametricAmplifier, PhaseMatchingType as OpaPhaseMatchingType, QuasiPhaseMatching,
};
pub use crate::temporal_photonics::photonic_time_crystal::{
    PhotonicTimeCrystal, SpatiotemporalCrystal,
};
pub use crate::temporal_photonics::time_refraction::{TemporalInterface, TimeSlab};

// Photoacoustics — PA generation, PAT imaging, optoacoustics, PA spectroscopy
pub use crate::photoacoustics::optoacoustic::{
    AcoustoOpticModulator, StimulatedBrillouinScattering, ThermalLensing,
};
pub use crate::photoacoustics::pa_generation::{
    GrueneisenParameter, PhotoacousticSource, SpectralUnmixing,
};
pub use crate::photoacoustics::pa_imaging::{
    back_projection_weight, CircularPetArray, DelayAndSumBeamformer, PatResolution,
    UniversalBackProjection,
};
pub use crate::photoacoustics::pa_spectroscopy::{
    beer_lambert_absorptance, ideal_gas_number_density, PaCell, PaGasSensor,
};

// Metamaterials — bulk negative-index media, transformation optics, EMT, hyperlens
pub use crate::metamaterials::effective_medium::{BruggemanEmt, MaxwellGarnett, MultilayerEmt};
pub use crate::metamaterials::hyperlens::{OpticalHyperlens, PendrySuperLens, SphericalSuperlens};
pub use crate::metamaterials::negative_index::{
    DoubleNegativeMedium, DrudeWireArray, SplitRingResonator,
};
pub use crate::metamaterials::transformation_optics::{
    CarpetCloak, CylindricalCloak, LuneburgLens, MaxwellFishEye,
};

// Photonic Antennas — nanoantenna, OPA, LiDAR, pattern analysis
pub use crate::photonic_antenna::{
    directivity_dbi_from_pattern,
    directivity_from_pattern,
    effective_aperture_m2,
    free_space_path_loss_db,
    friis_equation,
    gain_dbi_to_linear,
    gain_linear_to_dbi,
    // Radiation pattern utilities
    AntennaPatternMetrics,
    // Nanoantenna theory
    HertzianDipole,
    NanorodAntenna,
    // LiDAR OPA
    OpaLidar,
    // Optical phased arrays
    OpticalPhasedArray1d,
    OpticalPhasedArray2d,
    SiliconOpa,
    YagiUdaAntenna,
};

// PIC Simulation — circuit-level transfer-matrix & noise models
pub use crate::pic_simulation::{
    // Core types
    Complex as PicComplex,
    // Circuit elements
    DirectionalCoupler as PicDirectionalCoupler,
    // Cascade infrastructure
    GratingCoupler as PicGratingCoupler,
    MachZehnderInterferometer,
    MicroringResonator,
    MonteCarloYield,
    OsnrModel,
    PhaseNoise,
    PicCascade,
    PolarizationDependentLoss,
    // Yield and variability
    ProcessVariation,
    RinNoise,
    // Noise models
    ShotNoise,
    ThermalNoise,
    TransferMatrix2x2,
    TrimCorrection,
    WaveguideSection as PicWaveguideSection,
    YJunction,
    YieldModel,
};

// Optical CDMA
pub use crate::optical_cdma::{
    // Performance analysis
    erfc_approx as ocdma_erfc_approx,
    q_function as ocdma_q_function,
    // Transceivers & MAI
    CoherentOcdma,
    // Spreading codes
    GoldCode,
    IncoherentOcdma,
    MaiAnalyzer,
    MultipleAccessComparison,
    OokOcdmaBer,
    OpticalOrthogonalCode,
    OvsfTree,
    // Spectral encoding
    SacOcdma,
    SpcOcdma,
};
