## Canny Detector

What causes an edge?

1. Depth discontinuity
2. Surface orientation discontinuity
3. Surface color discontinuity
4. Illumination discontinuity

Edge: where pixel intensity changes drastically along one direction in the image, and almost no changes in the pixel intensity values along its orthogonal direction.
Jointly detecting edge and smoothing by convolving with the <u>derivative</u> of a Gaussian filter
$$
g = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{-\frac{x^2}{2\sigma^2}}
$$
Non-maximal suppression

Compare the central pixel with the gradient direction neighbors, keep it if is maximum

Thresholding and linking (hysteresis):

If the current pixel is not an edge, check the next one.
If it is an edge, check the two pixels in the direction of the edge (i.e., perpendicular to the gradient direction). If either of them (or both)
• have the direction in the same bin as the central pixel
• gradient magnitude is greater than minVal
• they are the maximum compared to their neighbors (NMS for these pixels),
mark these pixels as an edge pixel

Canny shows that the first derivative of the Gaussian closely approximates the operator that optimizes the product of signal-to-noise ratio and localization.

## Harris Detector

Corner response is equivariant with both translation and image rotation, not invariant to scale

- Image derivatives ——- to get $I_x$ and $I_y$
- Square of derivatives ——- to get $I_x^2, I_y^2, I_xI_y$
- Rectangle window or Gaussian filter ——- Gaussian filter could keep rotation invariant
- Then use corner response function ——- $\theta = g(I_x^2) g(I_y^2) - (g(I_xI_y))^2 - \alpha (g(I_x^2) + g(I_y^2))^2 - t$
- Non-maximum suppression

Along a direction $(u, v)$ we can get the intensity difference
$$
E_{(x, y)}(u, v) = g \star \left(I[x+u, y+v] - I[x, y]\right)^2 \approx \mqty[u, v] g \star \mqty[ I_x^2 & I_xI_y \\ I_xI_y & I_y^2] \mqty[u \\ v] = R^{-1}\mqty[\dmat{\lambda_1, \lambda_2}]R
$$
Since we want $1/k < \lambda_1/\lambda_2 < k$ and $\lambda_1, \lambda_2 > b$, we can use $\theta$ to present the constrain
$$
\theta = \frac{1}{2}\left(\lambda_1\lambda_2 - \alpha(\lambda_1+ \lambda_2)^2 \right) + \frac{1}{2}(\lambda_1\lambda_2 - 2t) = \lambda_1\lambda_2 - \alpha(\lambda_1+\lambda_2)^2 - t = \det{M} - \alpha (\Tr{M})^2 - t
$$
To be scale invariant, we can use Laplacian in scale or difference of Gaussians in space and scale

## Least Square Method

write line as $ax + by = d$, and $E = \sum (ax_i + by_i - d)^2 = Ah=0$, where $A$ is written from the data and $h$ is the model parameters, write $A = UDV^{\vb{T}}$, then $h$ is the last column of $V$

RANSAC: Single mode: robust for outliers

Hough Transform: Less robust compared to RANSAC, can handle multiple modes

## Gradient Descent

Batch Gradient Descent: Take all data and label pairs in the training set to calculate the gradient. -: very slow
-: easily get trapped at local minima

Stochastic Gradient Descent: Randomly sample N pairs as a batch from the training data and then compute the average gradient from them.

+: fast
+: can get out of local minima

## Activate function

ReLU

reduced likelihood of the gradient to vanish, not only won’t vanish when input increase, but also the maximum of the gradient is 1 so won’t drop to zero fast when the layer goes heavy; sparsity; quick

## Layer

Convolution

Sparse Connectivity; Parameter Sharing; Equivariance with Translation; reducing the “noise” in an image via filtering the image for important features; 

Pooling

invariance to small translations and rotations; drastically reduces the size of the image going into the next layer; enhance certain features, for example taking the max value we can further highlight the edges in an image whilst reducing its size and still keeping the main idea of the image.

## Initialization

Xavier: $\operatorname{var}{w_i} = 1/ D_{\text{in}}$ to keep the variance between input and output equal

He: $\operatorname{var}{w_i} = 2/D_{\text{in}}$ to keep variance under ReLU

## Optimization

SGD: loss may have high condition number, very slow progress along shallow dimension and jitter along steep direction; local minima or saddle point, zero gradient and get stuck; gradient can be noisy

learning rate: when increase the batch size by $N$, also scale the initial learning rate by $N$

-------



Batch Normalization: inserted after fully connected or convolutional layers, and before nonlinearity; keep feature to become equal and proceed smoothly down to the minimum, more robust to initialization; Scale and Shift are trainable parameters such that each Batch Norm layer is able to optimally find the best factors for itself, and can thus shift and scale the normalized values to get the best predictions

Skip link: promote flat minimizers and prevent the transition to chaotic behavior

Data augmentation: Scaling, Cropping, Flipping, Padding, Rotation, Translation, Affine transformation, Brightness, Contrast, Saturation, Hue; reducing data overfitting and creating variability in data; increasing generalization ability of the models; helping resolve class imbalance issues in classification

Dropout: Force the network to have a redundant representation, prevent co-adaptation of features; must scale the activation for each neuron

Bottleneck Residual Block: reduces the number of parameters and matrix multiplications. The idea is to make residual blocks as thin as possible to increase depth and have less parameters.

Skip link: makes shortcut from the inputs to the outputs; Assist final segmentation; Avoid memorization

Bottleneck: no need to memorize the whole image but only provides global context; Large receptive field and provides global context; Get rid of redundant information; Lower the computation cost

Architecture Drawback: as the depth of the encoder becomes deeper, the high-level feature of the original image is extracted, whereas the corresponding decoder block just started restoring. On the other hand, the earliest block of the encoder extracts the low-level feature, but the matching decoder block connected is the block closest to the prediction. In other words, U-Net connects low-level features close to the prediction layer and connects high-level features far to the prediction layer. This is an inevitable limit for a single set of encoder-decoder architecture.

# Calibrate camera

For a Point $\vb*{P}(x, y, z)$, to the camera, the imaging process is
$$
\vb*{P}' = \mqty[\alpha & -\alpha \cot \theta & c_x & 0 \\ 0 & \beta / \sin\theta & c_y & 0 \\ 0 & 0 & 1 & 0] \mqty[\vb{R} & \vb*{T} \\ 0 & 1]  \mqty[x \\ y \\ z \\ 1] = \vb{M}\vb*{P}_w
$$
where $\vb{R}_{3\times 3} = \vb{R}_x(\alpha)\vb{R}_y(\beta)\vb{R}_z(\gamma) = \mqty[\vb*{r}_1 & \vb*{r}_2 & \vb*{r}_3]^{\vb{T}},\ \vb*{T}_{3 \times 1} = \mqty[T_x & T_y & T_z]^{\vb{T}},\ \vb{M} = \mqty[\vb*{m}_1 & \vb*{m}_2 & \vb*{m}_3]^\vb{T}$

For the known information, we can get the equations as
$$
u_i(\vb*{m}_3 P_i) - \vb*{m}_1P_i = 0 \quad v_i(\vb*{m}_3P_i) - \vb*{m}_2P_i = 0
$$
With the SVD, we can get $\vb{M} = \mqty[\vb{A} & \vb*{b}]$ and $\vb{A} = \mqty[\vb*{a_1} & \vb*{a}_2 & \vb*{a}_3]^\vb{T}$, then
$$
\rho = \frac{\pm 1}{\abs{\vb*{a}_3}} \quad c_x = \rho^2(\vb*{a}_1\cdot \vb*{a}_3) \quad c_y = \frac{(\vb*{a}_1 \times \vb*{a}_3) \cdot (\vb*{a}_2 \times \vb*{a}_3)}{\abs{\vb*{a}_1 \times \vb*{a}_3}\abs{\vb*{a}_2 \times \vb*{a}_3}} \quad \alpha = \rho^2 \abs{\vb*{a}_1 \times \vb*{a}_3}\sin \theta \quad \beta = \rho^2\abs{\vb*{a}_2 \times \vb*{a}_3}\sin \theta
$$

$$
\vb*{r}_1 = \frac{\vb*{a}_2 \times \vb*{a}_3}{\abs{\vb*{a}_2 \times \vb*{a}_3}} \quad \vb*{r}_3 = \frac{\pm \vb*{a}_3}{\abs{\vb*{a}_3}} \quad \vb*{r}_2 = \vb*{r}_3\times \vb*{r}_1 \quad \vb*{T} = \rho \vb{K}^{-1}\vb*{b}
$$

From two view point, the disparity is
$$
u-u'=Bf/z
$$
$u$ is the first position to $O$ and $u'$ is the second position to $O'$ and $B$ is the distance between $\overline{OO'}$, $f$ is the focal length.

Challenge: Occlusions, Fore shortening, Brightness, Homogeneous regions, Repetitive patterns

![image-20230423183651337](C:\Users\zhuyu\Desktop\course\CVIntro\image-20230423183651337.png)

# Point Cloud

Uniform Sampling: sample $a_1$ and $a_2$ uniformly from $[0, 1]$, if $a_1 + a_2 \leqslant 1$, $x = a_1 v_1 + a_2v_2 + (1-a_1-a_2)v_3$, else $x = (1 - a_1)v_1 + (1-a_2)v_2 + (a_1 + a_2)v_3$

Normal Sampling: sample $r_1$ and $r_2$ from $U(0, 1)$, $x = (1-\sqrt{r_1})v_1 + \sqrt{r_1}(1-r_2)v_2 + \sqrt{r_1}r_2v_3$

Iterative Furthest Point Sampling: Over sample the shape by any fast method; iteratively select one particle with the largest distance to the selected point

Chamfer distance: $d_{\text{CD}}(S_1, S_2) = \sum_{x\in S_1}\min_{y\in S_2} \norm{x - y}_2 + \sum_{y\in S_2}\min_{x\in S_1} \norm{x - y}_2$; Sum of the closest distances; Insensitive to sampling

Earth Mover’s distance: $d_{\text{EMD}}(S_1, S_2) = \min_{\phi: S_1 \rightarrow S_2} \sum_{x\in S_1} \norm{x-\phi(x)}_2$; Sum of the matched closest distances; Sensitive to sampling

PointNet++: Recursively apply pointnet at local regions(farthest point sampling + grouping + pointnet); Hierarchical feature learning; Local translation invariance; Permutation invariance

# Voxel

### Sparse Conv

Pros: A way higher efficiency than dense conv; Regular grid that supports indexing; Similarly expressive compared to 2D Conv; Translation equivariance similar to 2D Conv
Cons: Discretization error

Sparse Conv:
+: Kernels are spatial anisotropic
+: More efficient for indexing and neighbor query
+：suitable for large-scale scenes
-: limited resolutions
Point cloud networks:
+: high resolution
+: easier to use and can be the first choice for a quick try
-: slightly lower performance
-: slower if performing FPS and ball query
