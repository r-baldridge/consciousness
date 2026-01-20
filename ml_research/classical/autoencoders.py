"""
Autoencoders

Research index entry for autoencoders, neural networks trained to
reconstruct their input through a bottleneck, learning compressed
representations (encoding).

Historical development:
- Early work on linear autoencoders (PCA equivalence)
- Nonlinear autoencoders for feature learning
- Sparse autoencoders (Ng, 2011)
- Denoising autoencoders (Vincent et al., 2008)
- Contractive autoencoders (Rifai et al., 2011)
- Variational autoencoders (Kingma & Welling, 2013)

Key contributions:
- Unsupervised feature learning
- Dimensionality reduction
- Generative modeling foundations
"""

from typing import Dict, List

from ..core.taxonomy import (
    MLMethod,
    MethodEra,
    MethodCategory,
    MethodLineage,
)


def get_method_info() -> MLMethod:
    """Return the MLMethod entry for Autoencoders."""
    return MLMethod(
        method_id="autoencoder_1986",
        name="Autoencoder",
        year=1986,
        era=MethodEra.CLASSICAL,
        category=MethodCategory.GENERATIVE,
        lineages=[MethodLineage.PERCEPTRON_LINE],
        authors=[
            "David E. Rumelhart",
            "Geoffrey E. Hinton",
            "Ronald J. Williams",
            "Hervé Bourlard",
            "Yves Kamp",
        ],
        paper_title="Learning Internal Representations by Error Propagation (Chapter 8, PDP Vol. 1)",
        paper_url="https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf",
        key_innovation="""
        Autoencoders learn to compress and reconstruct data:

        1. ENCODER-DECODER ARCHITECTURE: The network has a bottleneck
           layer (code/latent space) smaller than the input. The encoder
           compresses input to this representation, the decoder reconstructs.

        2. UNSUPERVISED LEARNING: Trained to minimize reconstruction error
           (input vs output), no labels needed. Learns useful features
           from the data distribution itself.

        3. FEATURE LEARNING: The bottleneck forces the network to learn
           the most salient features for reconstruction. These features
           often transfer well to downstream tasks.

        4. NONLINEAR PCA: Linear autoencoders learn the same subspace as
           PCA. Nonlinear activations enable more powerful representations.

        Key variants add regularization to the basic architecture:
        - Sparse: encourage few active neurons in code
        - Denoising: reconstruct clean data from corrupted input
        - Contractive: penalize sensitivity to input perturbations
        - Variational: probabilistic latent space with KL regularization
        """,
        mathematical_formulation="""
        BASIC AUTOENCODER:
            Encoder: z = f_enc(x) = sigma(W_enc * x + b_enc)
            Decoder: x_hat = f_dec(z) = sigma(W_dec * z + b_dec)

            Loss: L = ||x - x_hat||^2  (reconstruction error)

        DEEP AUTOENCODER:
            Encoder: h_1 = f(W_1 x + b_1), ..., z = f(W_L h_{L-1} + b_L)
            Decoder: symmetric (often tied weights: W_dec = W_enc^T)

        SPARSE AUTOENCODER:
            L = ||x - x_hat||^2 + beta * KL(rho || rho_hat)

            Where:
            rho_hat_j = (1/n) sum_i a_j(x_i)  (average activation of unit j)
            rho = target sparsity (e.g., 0.05)
            KL = sum_j rho log(rho/rho_hat_j) + (1-rho) log((1-rho)/(1-rho_hat_j))

        DENOISING AUTOENCODER:
            x_tilde = corrupt(x)  (add noise or mask)
            z = f_enc(x_tilde)
            x_hat = f_dec(z)
            L = ||x - x_hat||^2  (reconstruct clean from noisy!)

        CONTRACTIVE AUTOENCODER:
            L = ||x - x_hat||^2 + lambda * ||J_f(x)||_F^2

            Where J_f is the Jacobian of the encoder:
            ||J_f||_F^2 = sum_{ij} (dh_j/dx_i)^2

        VARIATIONAL AUTOENCODER (early concepts):
            z ~ q(z|x)  (probabilistic encoder)
            x ~ p(x|z)  (probabilistic decoder)
            L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        """,
        predecessors=["mlp_1986", "hebbian_learning", "pca"],
        successors=["vae_2013", "bert_masked_lm", "diffusion_models"],
        tags=[
            "unsupervised",
            "feature_learning",
            "dimensionality_reduction",
            "reconstruction",
            "encoder_decoder",
            "representation_learning",
        ],
        notes="""
        Autoencoders were important precursors to modern representation
        learning and generative models. Key developments:

        1. Pre-training (2006): Hinton used stacked autoencoders to
           pre-train deep networks, showing deep learning was possible.

        2. Denoising (2008): Vincent et al. showed that corrupting inputs
           leads to more robust features and prevents trivial solutions.

        3. VAE (2013): Kingma & Welling added probabilistic interpretation,
           enabling principled generative modeling.

        4. Modern uses:
           - Pre-training embeddings
           - Anomaly detection (high reconstruction error = anomaly)
           - Data compression
           - Generative modeling (VAE, VQ-VAE)
        """,
    )


def pseudocode() -> str:
    """Return pseudocode for autoencoder variants."""
    return """
    AUTOENCODER ALGORITHMS
    ======================

    BASIC AUTOENCODER:
        Architecture:
            encoder_layers = [input_dim, hidden_1, ..., latent_dim]
            decoder_layers = [latent_dim, ..., hidden_1, input_dim]

        def forward(x):
            # Encode
            h = x
            for layer in encoder_layers:
                h = activation(W @ h + b)
            z = h  # latent representation

            # Decode
            h = z
            for layer in decoder_layers:
                h = activation(W @ h + b)
            x_hat = h  # reconstruction

            return x_hat, z

        def loss(x, x_hat):
            return mean_squared_error(x, x_hat)
            # or: binary_cross_entropy(x, x_hat) for normalized inputs

        Training:
            for batch in data_loader:
                x_hat, z = forward(batch)
                L = loss(batch, x_hat)
                backprop(L)
                update_weights()

    SPARSE AUTOENCODER:
        hyperparameters:
            rho = 0.05        # target sparsity
            beta = 3.0        # sparsity penalty weight

        def forward_with_activations(x):
            # Same as basic, but return hidden activations
            h = x
            activations = []
            for layer in encoder_layers:
                h = activation(W @ h + b)
                activations.append(h)
            z = h
            # ... decode ...
            return x_hat, z, activations

        def sparsity_loss(activations):
            total_kl = 0
            for h in activations:
                rho_hat = mean(h, axis=batch)  # average activation per unit
                kl = rho * log(rho / rho_hat) + (1 - rho) * log((1 - rho) / (1 - rho_hat))
                total_kl += sum(kl)
            return total_kl

        def loss(x, x_hat, activations):
            return mse(x, x_hat) + beta * sparsity_loss(activations)

    DENOISING AUTOENCODER:
        hyperparameters:
            noise_type = 'gaussian' | 'masking' | 'salt_pepper'
            noise_level = 0.3  # e.g., probability of masking

        def corrupt(x, noise_type, noise_level):
            if noise_type == 'gaussian':
                return x + noise_level * randn_like(x)
            elif noise_type == 'masking':
                mask = bernoulli(1 - noise_level, size=x.shape)
                return x * mask
            elif noise_type == 'salt_pepper':
                mask = bernoulli(noise_level, size=x.shape)
                return where(mask, randint(0, 1), x)

        def forward(x):
            x_corrupted = corrupt(x)
            z = encode(x_corrupted)
            x_hat = decode(z)
            return x_hat  # compare to CLEAN x

        def loss(x_clean, x_hat):
            return mse(x_clean, x_hat)  # reconstruct clean from noisy!

    CONTRACTIVE AUTOENCODER:
        hyperparameters:
            lambda = 0.1  # contraction penalty weight

        def jacobian_penalty(x, h):
            # Compute ||J_f||_F^2 where J_f = dh/dx
            # For sigmoid: dh_j/dx_i = W_ij * h_j * (1 - h_j)
            # Efficient computation without full Jacobian:
            J_squared = sum((W * h * (1 - h))^2)
            return J_squared

        def loss(x, x_hat, h, W):
            return mse(x, x_hat) + lambda * jacobian_penalty(x, h)

    STACKED AUTOENCODER PRE-TRAINING (Hinton 2006):
        # Layer-wise pre-training

        layers = [784, 500, 250, 100, 30]  # example architecture

        def pretrain():
            input_data = training_data

            for i in range(len(layers) - 1):
                # Create shallow autoencoder for this layer
                ae = Autoencoder(layers[i], layers[i+1])

                # Train on current input
                ae.train(input_data)

                # Transform data for next layer
                input_data = ae.encode(input_data)

                # Save weights for full network
                pretrained_weights[i] = ae.weights

        def finetune():
            # Stack all layers
            full_network = build_from_pretrained(pretrained_weights)

            # Fine-tune with backprop on supervised task
            full_network.train_supervised(data, labels)
    """


def key_equations() -> Dict[str, str]:
    """Return dictionary of key equations for autoencoders."""
    return {
        # Basic autoencoder
        "encoder": "z = f_enc(x; theta_enc)",
        "decoder": "x_hat = f_dec(z; theta_dec)",
        "reconstruction_loss_mse": "L_rec = (1/n) sum ||x_i - x_hat_i||^2",
        "reconstruction_loss_bce": "L_rec = -sum [x log(x_hat) + (1-x) log(1-x_hat)]",
        # Tied weights
        "tied_weights": "W_dec = W_enc^T (transpose of encoder weights)",
        # Linear autoencoder = PCA
        "linear_ae_solution": "Linear autoencoder learns span of top k principal components",
        # Sparse autoencoder
        "average_activation": "rho_hat_j = (1/n) sum_i h_j(x_i)",
        "kl_sparsity": "KL(rho || rho_hat) = rho log(rho/rho_hat) + (1-rho) log((1-rho)/(1-rho_hat))",
        "sparse_loss": "L = L_rec + beta * sum_j KL(rho || rho_hat_j)",
        # Denoising autoencoder
        "denoising_objective": "L = E_{x, x_tilde} ||x - f_dec(f_enc(x_tilde))||^2",
        "masking_noise": "x_tilde = x * m where m_i ~ Bernoulli(1 - p)",
        "gaussian_noise": "x_tilde = x + epsilon where epsilon ~ N(0, sigma^2)",
        # Contractive autoencoder
        "jacobian_frobenius": "||J_f||_F^2 = sum_{ij} (dh_j/dx_i)^2",
        "contractive_loss": "L = L_rec + lambda * ||J_f(x)||_F^2",
        "jacobian_sigmoid": "(dh_j/dx_i) = W_ij * h_j * (1 - h_j)",
        # Variational (preview)
        "vae_elbo": "L = E_q[log p(x|z)] - KL(q(z|x) || p(z))",
        "reparameterization": "z = mu + sigma * epsilon where epsilon ~ N(0, I)",
    }


def get_autoencoder_variants() -> List[Dict[str, str]]:
    """Return detailed descriptions of autoencoder variants."""
    return [
        {
            "name": "Undercomplete Autoencoder",
            "description": """
                Bottleneck has fewer dimensions than input, forcing
                compression. Most common type.
            """,
            "latent_dim": "latent_dim < input_dim",
            "regularization": "Bottleneck itself is the regularizer",
        },
        {
            "name": "Overcomplete Autoencoder",
            "description": """
                Bottleneck has more dimensions than input. Requires
                additional regularization to prevent trivial identity mapping.
            """,
            "latent_dim": "latent_dim > input_dim",
            "regularization": "Sparsity, noise, or other penalties required",
        },
        {
            "name": "Sparse Autoencoder",
            "description": """
                Adds KL divergence penalty to encourage sparse activations
                in the hidden layer. Only a few units active for each input.
            """,
            "key_paper": "Ng (2011) - Sparse autoencoder tutorial",
            "benefit": "Learns more interpretable, disentangled features",
        },
        {
            "name": "Denoising Autoencoder (DAE)",
            "description": """
                Trained to reconstruct clean input from corrupted version.
                Corruption can be: masking, Gaussian noise, salt & pepper.
            """,
            "key_paper": "Vincent et al. (2008) - Extracting and Composing Robust Features",
            "benefit": "More robust features, learns data manifold structure",
        },
        {
            "name": "Contractive Autoencoder (CAE)",
            "description": """
                Adds penalty on the Frobenius norm of the encoder Jacobian,
                making the representation insensitive to small input changes.
            """,
            "key_paper": "Rifai et al. (2011) - Contractive Auto-Encoders",
            "benefit": "Learns locally invariant features, robust representations",
        },
        {
            "name": "Variational Autoencoder (VAE)",
            "description": """
                Probabilistic model with explicit prior on latent space.
                Encoder produces distribution parameters, not point estimate.
            """,
            "key_paper": "Kingma & Welling (2013) - Auto-Encoding Variational Bayes",
            "benefit": "Principled generative model, smooth latent space",
        },
        {
            "name": "VQ-VAE (Vector Quantized VAE)",
            "description": """
                Uses discrete latent codes via vector quantization.
                Learns a codebook of embeddings.
            """,
            "key_paper": "van den Oord et al. (2017)",
            "benefit": "Avoids posterior collapse, enables high-fidelity generation",
        },
        {
            "name": "Masked Autoencoder (MAE)",
            "description": """
                Masks large portions of input (e.g., 75%) and reconstructs.
                Highly effective for vision pre-training.
            """,
            "key_paper": "He et al. (2021) - Masked Autoencoders Are Scalable Vision Learners",
            "benefit": "Excellent pre-training for vision transformers",
        },
    ]


def get_applications() -> List[Dict[str, str]]:
    """Return common applications of autoencoders."""
    return [
        {
            "application": "Dimensionality Reduction",
            "description": """
                Nonlinear alternative to PCA. The latent code z provides
                a low-dimensional representation of high-dimensional data.
            """,
            "example": "Reduce images to 32D for visualization or clustering",
        },
        {
            "application": "Feature Learning / Pre-training",
            "description": """
                Learn useful representations without labels, then use
                encoder features for supervised tasks.
            """,
            "example": "Pre-train on unlabeled images, fine-tune for classification",
        },
        {
            "application": "Anomaly Detection",
            "description": """
                Train on normal data only. Anomalies have high reconstruction
                error because they differ from the training distribution.
            """,
            "example": "Detect fraudulent transactions, manufacturing defects",
        },
        {
            "application": "Denoising",
            "description": """
                Denoising autoencoders directly learn to remove noise.
                Can also denoise new images at test time.
            """,
            "example": "Image denoising, signal processing",
        },
        {
            "application": "Data Compression",
            "description": """
                Use encoder to compress, decoder to decompress. Can be
                more efficient than traditional codecs for specific domains.
            """,
            "example": "Learned image compression rivaling JPEG2000",
        },
        {
            "application": "Generative Modeling",
            "description": """
                VAEs and variants can generate new samples by sampling
                from the latent space and decoding.
            """,
            "example": "Generate new faces, molecules, music",
        },
        {
            "application": "Recommendation Systems",
            "description": """
                Encode user-item interaction matrices. Missing entries
                predicted from reconstruction.
            """,
            "example": "Netflix prize-style collaborative filtering",
        },
        {
            "application": "Semantic Hashing",
            "description": """
                Learn binary codes for efficient similarity search.
                Similar items have similar hash codes.
            """,
            "example": "Image retrieval, document search",
        },
    ]


def get_training_tips() -> Dict[str, str]:
    """Return practical training tips for autoencoders."""
    return {
        "architecture": """
            - Use symmetric encoder/decoder (same number of layers/units)
            - Tied weights can work but not always necessary
            - Start with small bottleneck, increase if underfitting
        """,
        "activation_functions": """
            - Hidden layers: ReLU or LeakyReLU work well
            - Output layer: depends on data
              - Sigmoid for [0,1] normalized images
              - Linear for unbounded real values
              - Tanh for [-1,1] normalized data
        """,
        "loss_function": """
            - MSE for real-valued continuous data
            - Binary cross-entropy for binary/normalized data
            - Custom perceptual losses for images (VGG features)
        """,
        "preventing_trivial_solutions": """
            - Use bottleneck smaller than input
            - Add noise (denoising autoencoder)
            - Add regularization (sparsity, contraction)
            - Dropout can help
        """,
        "monitoring": """
            - Watch reconstruction quality (visualize samples)
            - Monitor latent space structure (t-SNE/UMAP)
            - Check for mode collapse in VAEs
        """,
        "common_issues": """
            - Blurry reconstructions: try perceptual loss or adversarial loss
            - Posterior collapse (VAE): KL annealing, free bits
            - Overfitting: add regularization, more data, smaller model
        """,
    }
