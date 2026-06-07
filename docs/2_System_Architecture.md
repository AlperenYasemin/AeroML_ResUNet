# SECTION 2: System Architecture & Tech Stack

## 2.1. High-Level Architecture

The AeroML system is built on a decoupled, three-tier **client-server architecture** rather than a monolithic structure. Model weights and the PyTorch runtime environment are not embedded into the end-user's device. While all heavy matrix computations are isolated on a central server, clients only send lightweight JSON data and receive Base64-encoded PNG images in return.

This architectural decision ensures that the system remains platform-independent (Web, iOS, Android) and allows for instantaneous server-side model updates without requiring user-side intervention.

## 2.2. Backend & Inference Engine

This is the core layer responsible for the system's computational load. To meet production standards, it is wrapped in an asynchronous web framework rather than a simple Python script.

* **Deep Learning Framework:** PyTorch.
* **Web Server:** FastAPI with Uvicorn (ASGI).
* **Lifecycle:** Upon server startup (**cold start**), the `aero_resunet_v2_perfect.pth` weight file (approx. 30MB) is read from the disk and permanently loaded into RAM (or CUDA VRAM) and set to `eval()` mode. This eliminates I/O latency by preventing the model from reloading for every incoming HTTP request.
* **Bottleneck Analysis:** Deep learning models are inherently sequential on the GPU. While FastAPI accepts asynchronous requests, matrix multiplications for 100 simultaneous users must queue on the graphics card. To scale under high concurrency, future versions must export the model to C++ based engines like **ONNX** or **TensorRT**.

## 2.3. Frontend / Client Layer

This is the presentation layer where the user modifies boundary conditions and views live aerodynamic fields (pressure, velocity).

* **Framework:** React.js with State Management.
* **Data Format:** The client transmits wind velocities in the X and Y axes ($U_x$, $U_y$) and the selected wing profile to the server via a standard HTTP POST request body (JSON).
* **Optimization (Debouncing Filter):** Constant movement of the angle-of-attack slider poses a risk of triggering dozens of API requests per second, which could effectively perform a self-inflicted DDoS attack on the server. To prevent this, a **Debounce algorithm** is integrated into the interface. The system waits 150–200 milliseconds after the last slider movement; if the user has stopped, it sends a single request for the final values.

## 2.4. Hardware Requirements & Deployment

The promised real-time (millisecond-level) inference performance is strictly hardware-dependent.

* **Compute:** On CPU-only servers (e.g., standard AWS EC2 t3.micro), feature extraction for a $1 \times 3 \times 256 \times 256$ input tensor can take 1 to 2 seconds. To reach the **40ms latency goal**, the system must be hosted on a machine with at least an entry-level Tensor Core accelerator (e.g., Nvidia T4 or L4 GPU).
* **Containerization:** To avoid version conflicts between PyTorch, CUDA drivers, and FastAPI, the entire backend layer is packaged as a **Docker image**. This prevents "dependency hell" when moving code from the local development environment to a cloud server.