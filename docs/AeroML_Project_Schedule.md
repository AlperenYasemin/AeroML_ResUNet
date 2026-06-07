# AeroML System Development — Official Project Schedule

### Project Deadline: June 8, 2026

### Work

### Package

### Title Start Date End Date Deliverables / Tasks Status

### WP 1 Core Machine Learning Engine Mar 2 Mar 8

```
Preprocessing of the unstructured AirfRANS point cloud dataset (OpenFOAM/RANS
simulations) into structured 256×256 pixel tensors via SciPy griddata linear interpolation.
Generation of Signed Distance Field (SDF) geometric encoding for each airfoil. Normalization of
pressure (p) and velocity (Ux, Uy) fields to [-1, 1] via min-max scaling. Implementation of
AeroResUNet architecture: 4-stage Encoder (ResBlocks + MaxPool2d, 64→512 channels), 512-
channel Bottleneck, ConvTranspose2d Decoder with skip connections. Training for 100 epochs
using MSE loss, AdamW optimizer, and Cosine Annealing LR scheduler on CUDA. Baseline
inference validation targeting ~40ms latency per prediction.
```
```
Complete
```
### WP 2 Backend API Architecture & Documentation Mar 9 Mar 15

```
Development of production-grade FastAPI + Uvicorn (ASGI) server wrapping the PyTorch
inference engine. Cold-start model loading: aero_resunet_v2_perfect.pth (~30MB) loaded into
RAM/VRAM at boot and locked in eval() mode to eliminate per-request I/O latency.
Configuration of POST /predict endpoint accepting JSON payload (ux_in, uy_in boundary
conditions). Server-side heatmap rendering pipeline: raw tensor output → colored heatmap →
Base64-encoded PNG → JSON response, offloading all rendering from client devices. Stateless
REST API design (no session/prediction history stored). Formal system documentation covering
API schema, architecture decisions, and known bottlenecks (GPU concurrency limits, future
ONNX/TensorRT migration path).
```
```
Complete
```
### WP 3 Web Frontend Prototyping Mar 16 Mar 22

```
React.js project initialization and component architecture planning. UI/UX design for the
aerodynamic control panel: sliders for Angle of Attack (α, range -5° to +15°) and free-stream
velocity (U∞), with live heatmap canvas for pressure (p) and velocity (Ux, Uy) fields. State
management setup to track boundary condition inputs and API response data. Basic component
rendering and layout validation across desktop viewports.
```
```
Complete
```
### WP 4

```
Frontend-Backend Integration &
Optimization
```
```
Mar 23 Mar 29
```
```
Establishing HTTP REST communication between React.js frontend and FastAPI backend via
POST /predict with JSON payload (ux_in, uy_in). Implementation of Debounce algorithm (150–
200ms threshold) on slider inputs to prevent DDoS-like API flooding from rapid user interactions.
Rendering pipeline: decoding Base64-encoded PNG heatmap responses and painting them to
the web canvas targeting 20 FPS live update rate. End-to-end latency profiling and optimization
to validate the ~40ms inference target on GPU-backed server.
```
```
Current
```
### WP 5 Mobile Client Initialization Mar 30 Apr 5

```
React Native environment setup for cross-platform iOS/Android deployment. Adaptation of
control panel UI (AoA and velocity sliders) for mobile touch viewports. Cross-platform Base
PNG heatmap rendering on mobile canvas components. Preliminary API connection testing
targeting low-latency performance on cellular networks.
```
```
Upcoming
```
### BREAK

```
Midterm Break — Suspension of
Development
```
```
Apr 6 Apr 19 No project tasks scheduled due to academic examinations. —
```
### WP 6 Mobile Application Refinement Apr 20 Apr 26

```
Finalization of mobile touch controls with smooth gesture handling for AoA and velocity sliders.
State synchronization between mobile UI and backend inference results. Optimization of Base
heatmap decoding and canvas rendering for latency-free display on cellular networks. Cross-
device testing on iOS and Android to ensure consistent aerodynamic field visualization.
```
```
Upcoming
```
### WP 7

```
System Containerization & Deployment
Preparation
```
```
Apr 27 May 3
```
```
Authoring Dockerfiles for the FastAPI + Uvicorn backend, bundling Python 3.12, PyTorch
(CUDA-enabled), and all dependencies to eliminate environment-specific dependency conflicts
(dependency hell). Configuration of NVIDIA Container Toolkit for GPU hardware passthrough
into the Docker container to maintain 40ms inference performance in production. Resolution of
PyTorch/CUDA driver version conflicts between development and production environments.
Validation of containerized inference pipeline under simulated production conditions.
```
```
Upcoming
```
### WP 8 Cloud Deployment & CI/CD May 4 May 10

```
Deployment of GPU-enabled Docker backend to cloud instance with Tensor Core accelerator
(e.g., AWS/GCP with Nvidia T4 or L4 GPU) — required to meet 40ms latency vs. 1–2s on CPU-
only instances. Deployment of React.js frontend web application to static hosting. Configuration
of CORS policies, domain routing, and HTTPS for secure client-server communication. Post-
deployment latency benchmarking and concurrency stress testing.
```
```
Upcoming
```
### WP 9

```
Academic Enhancement (Physics-Informed
Feasibility)
```
```
May 11 May 17
```
```
Research into Physics-Informed Neural Networks (PINNs): integrating Navier-Stokes PDEs
(conservation of mass ∇·u=0 and momentum ρ(u·∇)u = −∇p + μ∇²u) as penalty terms in the
MSE loss function. Analysis of the current system's core limitation: purely statistical pixel-
matching with no mathematical guarantee of mass/momentum conservation. Theoretical
framework documentation for replacing ConvTranspose2d with Bilinear Upsampling to eliminate
checkerboard artifacts in decoder outputs. Feasibility assessment for future transition to spatio-
temporal 3D transient flow modeling.
```
```
Upcoming
```
### WP 10

```
Comprehensive Quality Assurance (QA) &
Bug Fixing
```
```
May 18 May 24
```
```
End-to-end stress testing of the full system stack: React/React Native clients → POST /predict
→ FastAPI/PyTorch inference → Base64 heatmap response. Concurrency load testing:
simulating 50+ simultaneous slider interactions to assess GPU queue bottlenecks and evaluate
need for Celery/RabbitMQ message broker integration. UI regression testing: identifying canvas
rendering glitches, debounce edge cases, and mobile viewport issues. API latency profiling
under concurrent requests and code refactoring for production readiness.
```
```
Upcoming
```
### WP 11

```
Academic Presentation & Repository
Polishing
```
```
May 25 May 31
```
```
Finalizing GitHub repository structure: organized modules for ML pipeline, FastAPI backend,
React web frontend, and React Native mobile client. Creation of architecture diagrams: three-tier
client-server topology, ResUNet encoder-decoder structure, and data preprocessing pipeline
(point cloud → SDF → tensor). Preparation of formal academic presentation and project defense
report covering: problem statement (CFD bottleneck), AeroML solution, ML pipeline, system
architecture, deployment strategy, and future PINN roadmap.
```
```
Upcoming
```
### WP 12 Final Review & Delivery Jun 1 Jun 7

```
Final code freeze across all system components (ML engine, FastAPI backend, web and mobile
frontends). Complete end-to-end system walkthrough: verifying inference latency, heatmap
rendering accuracy, mobile/web UI functionality, and Docker deployment integrity. Submission
preparation and final delivery ahead of June 8, 2026 project deadline.
```
```
Upcoming
```

## Category Colour Legend

### Colour Category Work Packages

### ML Development WP 1, WP 9

### Backend / API WP 2, WP 7, WP 8

### Web Frontend WP 3, WP 4

### Mobile WP 5, WP 6

### QA & Delivery WP 10, WP 11, WP 12

### Break Midterm Break


