# SECTION 3: ML Pipeline & Data Preprocessing

## 3.1. Dataset Topology: AirfRANS

The foundation of this project is the **AirfRANS** dataset, which consists of 2D airfoil simulations solved using RANS (Reynolds-Averaged Navier-Stokes) equations via OpenFOAM. The raw dataset is an **unstructured point cloud** rather than an image grid; it contains the spatial coordinates (X, Y) of each node and the associated scalar/vector fluid properties at that point.

Because standard Convolutional Neural Networks (CNNs) cannot operate directly on unstructured data, transforming this into a **structured Euclidean space** is a mandatory engineering step before the data enters the training pipeline.

## 3.2. Data Preprocessing and Spatial Transformation

The transition from a point cloud to a homogeneous $256 \times 256$ pixel tensor grid is the most critical stage determining the model's performance.

* **Interpolation:** Using the `griddata` function from the SciPy library, the unstructured Navier-Stokes solution points were mapped onto a regular $256 \times 256$ Cartesian grid via linear interpolation. During this process, loss of data in high-gradient regions—such as the **leading edge** and **trailing edge**—is inevitable. Reducing mesh density to pixel density represents the first deterministic error margin accepted by the system.
* **Geometric Encoding (Signed Distance Field - SDF):** To allow the model to mathematically perceive the wing surface (solid boundary), a **Signed Distance Field (SDF)** is used instead of a binary mask. Each pixel in the SDF matrix stores the shortest Euclidean distance to the wing surface. Values are negative inside the wing and positive outside, enabling the model to learn the gradient of the solid boundary.
* **Tensor Format:** Before entering the PyTorch training loop, data is standardized into the `[Batch, Channel, Height, Width]` format.
* **Input (X) $[N, 3, 256, 256]$:** Channel 0 (SDF), Channel 1 ($U_x$ free-stream), Channel 2 ($U_y$ free-stream).
* **Target (Y) $[N, 3, 256, 256]$:** Channel 0 (Pressure $p$), Channel 1 ($U_x$ velocity field), Channel 2 ($U_y$ velocity field).


* **Normalization:** To prevent numerical variance in pressure and velocity fields from causing **exploding gradients**, all feature maps are normalized to a narrow range (e.g., -1 to 1) using min-max scaling.

## 3.3. Network Architecture: AeroResUNet Synthesis

While a standard U-Net architecture is often sufficient for image-to-image translation, capturing high-frequency fluid details requires a deeper network. The **vanishing gradient** problem inherent in deep networks is addressed using **Residual Blocks (ResBlocks)**.

* **Encoder (Contracting Path):** Consists of ResBlocks with $3 \times 3$ convolutions and Batch Normalization. Spatial dimensions are halved ($256 \rightarrow 128 \rightarrow 64 \rightarrow 32$) using `MaxPool2d`, while the number of channels (features) increases ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$).
* **Bottleneck:** The lowest layer where the 512-channel feature matrix is most dense and spatial information is most compressed.
* **Decoder (Expanding Path):** Uses `ConvTranspose2d` (Deconvolution) layers to step-wise restore the spatial resolution back to $256 \times 256$.
* **Skip Connections:** To ensure microscopic details like **flow separation** and **boundary layers** are not lost during contraction, feature maps from Encoder layers are concatenated directly to their corresponding Decoder layers. Without these connections, the model would produce only blurry, generalized pressure distributions.

## 3.4. Optimization and Training Strategy

* **Loss Function:** The model is trained using **Mean Squared Error (MSE)**, which calculates the statistical deviation between the target CFD data and the model’s prediction. **Critical Warning:** Using MSE means the model is entirely unaware of physical laws (Navier-Stokes). It does not account for the conservation of mass or momentum; it simply attempts to minimize the "color" mismatch between pixels.
* **Optimizer:** The **AdamW** algorithm was chosen for its more stable application of L2 Regularization, which helps suppress overfitting.
* **Learning Rate Scheduling:** A **Cosine Annealing** scheduler was implemented to prevent the model from getting stuck in local minima and to ensure smoother convergence toward the end of the training process.
* **Training Loop:** The model was trained for 100 epochs on a hardware accelerator (CUDA), seeing the entire dataset in each epoch.