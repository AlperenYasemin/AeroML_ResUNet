# SECTION 1: Project Overview, Problem Statement, and Scope

## 1.1. Problem Statement

In the aerospace, automotive, and wind energy industries, aerodynamic design optimization traditionally relies on **Computational Fluid Dynamics (CFD)** solvers (e.g., OpenFOAM, ANSYS Fluent). CFD software numerically solves the incompressible **Navier-Stokes equations** to model fluid behavior:

**Conservation of Mass (Continuity):**


$$\nabla \cdot \mathbf{u} = 0$$

**Conservation of Momentum:**


$$\rho (\mathbf{u} \cdot \nabla)\mathbf{u} = - \nabla p + \mu \nabla^2 \mathbf{u}$$

Solving these equations (specifically using RANS - Reynolds-Averaged Navier-Stokes-based turbulence models) requires extremely fine computational meshes and high iteration counts. Even a simple 2D airfoil analysis at a single angle of attack can take minutes or hours on modern CPUs. This computational cost creates a significant bottleneck, restricting "rapid prototyping" and "real-time exploration" during the early stages of design.

## 1.2. Proposed Solution

**AeroML** is a data-driven surrogate model system developed to bypass this computational bottleneck. Instead of solving the Navier-Stokes equations from scratch every time, the project utilizes a **Deep Residual Network (ResUNet)** that learns spatial patterns from thousands of pre-solved CFD simulations (derived from the *AirfRANS* dataset).

The system directly injects target boundary conditions (free-stream velocity $U_{\infty}$ and angle of attack $\alpha$) and the **Signed Distance Field (SDF)** of the wing geometry into the model's input tensors. Using these inputs, the model predicts the normalized pressure ($p$) and velocity fields ($U_x$, $U_y$) around the wing in approximately **40 milliseconds**. Compared to traditional CFD, this represents a speedup of $10^4$ to $10^5$ times in computation time.

## 1.3. Scope and Limitations

To remain engineering-realistic, it must be noted that this system is not a direct replacement or a backup for traditional CFD solvers. It is a guidance tool for the initial stages of design. The rigid constraints of the system are as follows:

* **Dimensional Limit:** The model operates only on 2D sections. It cannot model 3D effects such as wingtip vortices.
* **Flow Regime:** The system is trained solely on steady-state RANS data. Transient flow separations or vortex shedding cannot be predicted with this architecture.
* **Physical Violations:** The model architecture is based entirely on statistical pixel mapping (Mean Squared Error). Navier-Stokes equations are not embedded into the system as a loss function (as seen in PINNs - Physics-Informed Neural Networks). Consequently, there is no mathematical guarantee that the model's output satisfies the conservation of mass and momentum with 100% accuracy.
* **Interpolation Limit:** The model performs with high accuracy within the Reynolds numbers, NACA profile types, and angle of attack ranges (-5° to +15°) covered by the AirfRANS dataset. In extreme conditions outside these bounds (extrapolation), the model's error margin (divergence) will increase rapidly.

## 1.4. Target Audience and Use Cases

The millisecond-level inference time of the system enables use cases that were previously impossible:

1. **Real-Time Web/Mobile Interaction:** Users can adjust wind speed or angle via sliders on an interface and watch aerodynamic results live at 20 frames per second (FPS).
2. **Conceptual Design Optimization:** An optimization algorithm (e.g., Genetic Algorithm) can converge on an optimum design by testing thousands of different geometries through AeroML in seconds.
