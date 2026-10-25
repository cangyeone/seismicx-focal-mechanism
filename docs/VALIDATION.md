# Validation evidence

[English](VALIDATION.md) · [简体中文](VALIDATION.zh-CN.md) · [Usage](../README.md)

Recorded run: 2026-10-25; Linux/WSL, two RTX 5090 GPUs, PyTorch 2.8.0+cu128, GRTM 0.6.0.dev12. The source binaries are external dependencies. Model identities are in [assets/manifest.json](../assets/manifest.json).

## Independent body-wave holdouts

Per Earth model, labels were generated at 32,768 training, 2,048 validation and 2,048 independent test points. Sobol scrambles differ between splits. The labels contain the earliest returned ray within each supported phase, travel-time input gradients, takeoff angle, source velocity and alternate-ray gap. Time/gradient supervision and phase-balanced experts use only the training split; checkpoints use internal validation.

The full-domain test spans 0–200 km source depth and 0–3000 km epicentral distance. It retains branch-switch tails. The table below is **mean absolute travel-time error in seconds**, not an upper bound. The existing first-P checkpoint is reused and is not included as a newly trained row.

| Phase / 震相 | iasp91 | ak135 | PREM |
|---|---:|---:|---:|
| S_first | 0.0542 | 0.0514 | 0.0455 |
| P | 0.0518 | 0.0536 | 0.2382 |
| p | 1.1503 | 1.1578 | 0.5658 |
| Pn | 0.0017 | 0.0017 | 0.0018 |
| Pg | 0.0089 | 0.0064 | 0.0998 |
| S | 0.0923 | 0.0928 | 0.3027 |
| s | 1.0763 | 0.9759 | 0.7835 |
| Sn | 0.0026 | 0.0024 | 0.0028 |
| Sg | 0.0131 | 0.0114 | 0.2123 |
| pP | 0.2456 | 0.2394 | 0.2527 |
| sP | 0.0467 | 0.0410 | 0.0473 |
| PP | 0.1729 | 0.1837 | 1.2395 |
| SS | 0.3701 | 0.4147 | 1.6347 |

[Full CSV](../verification/body_phase_accuracy.csv) reports time/angle MAE, P95, maximum, sample count, and the depth <=60 km / distance <=500 km regional subset separately. [Shared-network evidence](../verification/body_training.json) includes phase-availability precision/recall; [expert evidence](../verification/body_expert_training.json) reports final time/angle errors. The neural availability mask is approximate.

Large tails remain in some upgoing, reflected and multipath branches. Smooth DNNs cannot exactly reproduce discontinuous first-arrival gradients. These results do not establish a complete high-accuracy replacement for arbitrary TauP phases or ray multiplicities.

## Surface-wave holdouts

Three fixed flat-layer Earth models, Rayleigh/Love mode 0, independent log-period jitters over 2–80 s: 512 training, 128 validation and 128 test periods per Earth/wave family. Training-only target statistics are saved in the model. Missing modes retain masks rather than imputed velocities.

Across the six family holdouts, mean group-speed errors are approximately 0.0007–0.0009 km/s. [Detailed phase/group velocity errors](../verification/surface_training.json) include maxima. Higher modes and arbitrary new velocity-model families are outside this trained model's scope.

## Matched-model synthetic mechanism recovery

Physical GRTM observations use strike 35°, dip 50°, rake −70°, Mw 4, depth 10 km, eight stations, 1024 samples at 0.25 s, and 0.5% global-RMS Gaussian noise. The first-motion targets were generated from the known mechanism and neural rays.

Recovered depth: **10.0 km** among 8/10/12 km; Mw **3.999998**; normalized tensor angle **0.005147°**; weighted waveform VR **99.99949%**. A tensor angle is not a Kagan angle. Auxiliary-plane selection can change the displayed SDR while leaving the tensor equivalent.

[Result JSON](../verification/synthetic_result.json) · [Waveform plot](figures/waveform_fit.png) · [Mechanism plot](figures/mechanism.png). This is a matched-model recovery check, not real-event accuracy or evidence that the Earth model is correct.

## Real-data interface exercise

Event BJ.202202030155.0001: 45 raw-Z stations, CNN/manual first-motion agreement 44/45, and 4 polarity mismatches for the fitted first-motion mechanism. Catalog Pg was mapped to the explicit regional ray definition. Manual labels were used for auditing, not as CNN inputs. No authoritative mechanism truth is available. The raw records are not distributed in Git.

## Numerical gradient and quadrature checks

- [Pipeline audit](../verification/pipeline_validation.json): ObsPy/TauP imports blocked; DNN takeoff-angle second-derivative chains finite; native GRTM and the linear source bank agree at approximately 1e-14 waveform relative L2 and 1e-15 source-angle-gradient relative error in the recorded case.
- [Complete objective audit](../verification/joint_gradient_validation.json): central differences check strike, dip, rake, log amplitude, Pnl shift, Rayleigh shift and Love shift; recorded relative errors are below 2e-9.
- [Continuous input audit](../verification/continuous_input_smoke.json): picked and DNN-centered continuous crops reproduce the direct 1024-point input probabilities.
- [Quadrature audit](../verification/green_quadrature.json): spacing refinement changes body/surface filtered L2 by about 0.017%/0.042%; extending the high-wavenumber cutoff changes them by about 0.846%/1.854%. The recorded acceptance threshold is 2%, only for this specific regional case. It is not a universal integration certificate.
- Nine scientific unit checks passed before publication: source convention, auxiliary-plane equivalence, neural gradients, CNN input gradients, interpolation derivatives, filtering/rotation amplitude properties and synthetic first-motion recovery. [Clean-directory publication smoke tests](../verification/publication_smoke.json) also passed editable installation, asset checks, the console entry point, travel-time queries, first-motion inversion and the same nine unit tests.

Native GRTM geometry is fixed in its differentiable operator. Default depth estimation uses a physical outer grid with identical observed windows/weights at every depth. Optional piecewise-linear bank interpolation is approximate. There is no claim of exact native source-depth derivatives or continuously conditioned joint Earth-structure inversion.
