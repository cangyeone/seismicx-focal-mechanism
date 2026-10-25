# Provenance and third-party components

This private workflow reuses the owner's differentiable first-P TravelTimeMLP
implementation and packaged iasp91/ak135/PREM weights from the supplied
travel_time_mlp_local project. At the owner's request, the combined application,
weights and documentation are versioned in the private
cangyeone/seismicx-focal-mechanism repository. This is not authorization to
redistribute them publicly. The existing Research and Noncommercial License
and attribution remain in effect; no new open-source licence is granted.

The initial-motion CNN is the CSNDATA model trained in
csndata_polarity_cnn_20261024, best epoch 17. Its residual architecture was adapted
from cangyeone/seismological-ai-tools, commit
69f5061d8ee18f57cbbe38449f029b0a5b592849, polar.jit.py; packaging includes the
validated raw-Z normalization wrapper. The model is distributed here with its
existing provenance, not claimed as an independently invented architecture.

SeismicX GRTM is a separately installed research/noncommercial dependency.
No solver implementation source or binary is copied into this package. Its
documented NED moment-tensor, SI displacement and source-history conventions
are retained. Standard radial Earth models are approximated by flat solid
layers for the regional GRTM calculation; this approximation is explicit.

Offline body labels use separately installed ObsPy TauP (LGPL-3.0) and standard
iasp91/ak135/PREM velocity models, preserving their upstream rights. No modified
ObsPy source is embedded. Runtime does not import ObsPy or TauP. Surface labels
come from GRTM's fundamental Rayleigh/Love eigenmode solver. These neural
networks are not a complete arbitrary-phase or multimode replacement for TauP.

CAP follows the segmentation, amplitude-preserving residual and independent
time-shift principles of Zhu & Helmberger (1996), DOI 10.1785/BSSA0860051634.
This Torch implementation is not the authors' original CAP executable and
does not claim numerical identity with every configuration of that program.

NumPy, PyTorch, SciPy and Matplotlib are separately installed dependencies.
CSNDATA sources are read-only; derived windows and synthetic tests are kept
in the private project directories. Synthetic recovery is not a claim of
ground-truth accuracy for the real example earthquake.
