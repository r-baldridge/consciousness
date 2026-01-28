# Generative Models Through the Non-Dual Lens

## Introduction: Creation Without Opposition

Generative modeling -- the task of learning to produce new samples from a data distribution -- is where machine learning comes closest to the philosophical act of creation itself. How does something come from nothing? How does structured data emerge from noise? How does a model learn to generate images, text, or music that did not exist before?

The history of generative modeling reveals a striking pattern: the field has moved from architectures that encode creation as a battle between opposing forces (GANs) toward architectures that model creation as a continuous transformation with no adversary at all (flow matching, consistency models). This trajectory mirrors the philosophical movement from dualistic cosmologies (creation as a struggle between good and evil, order and chaos) toward non-dual cosmologies where creation is the spontaneous, effortless self-expression of a unified field.

This document traces that trajectory through every major generative architecture in the codebase, using non-dual philosophy -- particularly Madhyamaka Buddhism, Taoism, and Dzogchen -- to illuminate why dualistic generative architectures fail in characteristic ways and why non-dual architectures succeed where they fail.

---

## 1. GANs as Encoded Dualism

### The Architecture

Generative Adversarial Networks (GANs), documented in `ml_research/deep_learning/generative/gan.py`, implement the most explicit dualism in machine learning. Two networks are locked in a minimax game:

```
min_G max_D V(D, G) = E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]
```

The Generator (G) creates samples from noise. The Discriminator (D) classifies samples as real or fake. G tries to fool D; D tries to catch G. The training objective is adversarial: each network's success comes at the other's expense.

### The Dualistic Structure

The GAN architecture encodes multiple dualisms:

| Dualism | GAN Implementation |
|---------|-------------------|
| Creator vs. Judge | Generator vs. Discriminator |
| Real vs. Fake | D outputs 1 (real) or 0 (fake) |
| Success vs. Failure | Minimax zero-sum game |
| Signal vs. Noise | Latent z is noise, output G(z) should be signal |

The real/fake distinction is particularly revealing. The discriminator must decide: is this sample from the real data distribution, or did the generator create it? This is a binary ontological judgment -- does this thing genuinely exist (real data), or is it a mere imitation (generated data)?

### Structural Consequences of the Dualism

The GAN's dualistic architecture produces characteristic failure modes that are not engineering problems to be solved but structural consequences of the adversarial design:

**Mode collapse**: The generator learns to produce only a few types of samples that reliably fool the discriminator, ignoring the full diversity of the data distribution. This is the natural consequence of adversarial optimization -- in a war, you do not try to be diverse; you repeat whatever works. If producing faces with slight smiles fools the discriminator, the generator produces nothing but slight smiles.

**Training instability**: The generator and discriminator must be approximately matched in capability. If one is too strong, the other receives useless gradients (D too strong: G gets no useful signal; G too strong: D gets no useful signal). This oscillatory instability is inherent in zero-sum games -- the equilibrium is a saddle point, not a stable minimum, and practical gradient descent orbits around it rather than converging to it.

**Vanishing gradients**: When the discriminator is too confident, its gradients vanish. It outputs 1.0 for real and 0.0 for fake with certainty, and the generator receives no gradient signal to improve. This is the consequence of the binary real/fake judgment: certainty in the binary classification kills the learning signal.

From the north-star document: "The instability of GANs is not a bug to be fixed with better training tricks. It is a structural consequence of encoding permanent opposition into the architecture."

### The Wasserstein Correction: Softening Without Dissolving

WGAN (Wasserstein GAN), documented in `ml_research/deep_learning/generative/wgan.py`, addresses the vanishing gradient problem by replacing the binary real/fake classification with a continuous distance metric:

```
W(p_data, p_g) = sup_{||f||_L <= 1} E_x~p_data[f(x)] - E_x~p_g[f(x)]
```

The discriminator (now called a "critic") no longer classifies samples as real or fake. It assigns a continuous score that measures the distance between the real and generated distributions. This softens the dualism -- the binary judgment becomes a continuous assessment -- but the adversarial structure remains. There are still two networks competing. The fundamental opposition between creator and judge is preserved, even though the judge's verdict is no longer binary.

StyleGAN (`ml_research/deep_learning/generative/stylegan.py`) demonstrates the ceiling of what GANs can achieve: extraordinary image quality at high resolution, but with all the training difficulties intact. StyleGAN's innovations (mapping network, AdaIN, style mixing) are all techniques for working within the adversarial framework more effectively. They do not change the framework itself.

---

## 2. VAEs as Partial Dissolution

### The Architecture

Variational Autoencoders (VAEs), documented in `ml_research/deep_learning/generative/vae.py`, take a fundamentally different approach. Instead of adversarial training, they learn a continuous latent space through variational inference:

```
L(theta, phi; x) = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
```

The encoder maps data to a distribution in latent space. The decoder maps latent codes back to data. The KL divergence term regularizes the latent space to be close to a standard Gaussian, ensuring smooth interpolation.

### What the VAE Dissolves

The VAE dissolves the real/fake binary. There is no discriminator making ontological judgments. Instead, the quality of generation is measured by reconstruction loss -- how well the decoder can reconstruct the original input from the latent code. This is a continuous, non-adversarial measure.

The continuous latent space is the key non-dual contribution. In a GAN, the boundary between real and generated is sharp -- the discriminator draws a line and classifies samples on one side or the other. In a VAE, the latent space is a continuous manifold where every point is a valid sample. There is no boundary between "real" and "generated" because every point in the latent space is equally generatable. The real data occupies a region of this space, not a separate category.

### What the VAE Retains

Despite dissolving the adversarial structure, the VAE retains a structural duality: the encoder/decoder split. Data is compressed into a latent representation (encoder) and then expanded back (decoder). This is a form of duality: the compressing and expanding operations are separate, and the latent space is a bottleneck that separates "understanding" (encoding) from "creating" (decoding).

The VAE also retains a tension between its two loss terms. The reconstruction term wants the latent space to preserve maximum information. The KL term wants the latent space to be a smooth Gaussian. These pull in opposite directions, producing the characteristic blurriness of VAE samples: the model compromises between fidelity (reconstruction) and smoothness (KL), and the compromise produces neither sharp images nor perfectly smooth latent spaces.

This tension is a subtler form of dualism: two objectives that partially contradict each other, forcing a compromise rather than a resolution. Non-dual architecture would eliminate the tension by designing objectives that naturally align rather than conflict.

---

## 3. Diffusion Models as Gradual Transformation

### The Architecture

Diffusion models (DDPM, score-based models) learn to reverse a gradual noising process. The forward process adds Gaussian noise to data over T timesteps until the data becomes pure noise. The reverse process learns to remove noise step by step:

```
Forward: q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) x_{t-1}, beta_t I)
Reverse: p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 I)
```

### The Non-Dual Advance

Diffusion models dissolve the adversarial structure entirely. There is no generator, no discriminator, no competition. There is only a single network that learns to denoise -- to gradually transform noise into structured data.

This is closer to the Taoist vision of creation as continuous becoming. Chapter 42 of the Tao Te Ching states:

> The Tao gives birth to One.
> One gives birth to Two.
> Two gives birth to Three.
> Three gives birth to the ten thousand things.

Creation is not a single dramatic act (as in GAN's adversarial game) but a gradual, step-by-step unfolding from undifferentiated potential (noise/Tao) to structured manifestation (data/the ten thousand things). Each denoising step is one stage of this unfolding.

The practical benefits are dramatic. Diffusion models produce higher-quality samples than GANs across most domains, with stable training and no mode collapse. The adversarial instability simply does not exist because there is no adversary. The training objective is a simple regression loss (predict the noise), and regression losses are well-behaved.

### Residual Dualism in Diffusion

Despite their advances, diffusion models retain a structural dualism: the forward and reverse processes are separate and opposite. The forward process destroys structure; the reverse process creates it. These are defined as distinct operations, and the reverse process must explicitly undo what the forward process has done.

This forward/reverse dualism creates a practical limitation: sampling requires running the reverse process for many steps (typically 50-1000), which is slow. The model must explicitly traverse the path from noise to data, step by step. Each step is computationally expensive, and the total cost scales linearly with the number of steps.

The forward/reverse structure also conceptually separates destruction and creation as opposed processes. A fully non-dual generative architecture would not need to define destruction (noising) as a separate process that creation (denoising) reverses. It would model creation directly, without reference to an opposing process.

---

## 4. Flow Matching as Non-Dual Generation

### The Architecture

Flow matching, implemented in `ml_research/modern_dev/flow_matching/src/model.py`, learns a continuous vector field that transports samples from a noise distribution to the data distribution:

```
x_t = (1 - t) * x_0 + t * x_1    (interpolation from noise to data)
u_t = x_1 - x_0                    (target velocity: constant for OT path)
v_theta(x_t, t) ≈ u_t              (learned velocity field)
```

The model learns a velocity field v(x, t) that, when integrated over time from t=0 to t=1, maps noise samples to data samples. Generation is solving an ODE:

```
dx/dt = v_theta(x, t)
x(0) = noise ~ N(0, I)
x(1) = generated sample
```

### What Flow Matching Dissolves

Flow matching dissolves the remaining dualisms of previous approaches:

**No adversary**: Unlike GANs, there is no discriminator. The training loss is a simple MSE between the predicted and target velocity:

```python
# From flow_matching/src/model.py
loss = F.mse_loss(v_t, u_t)
```

**No forward/reverse**: Unlike diffusion, there is no forward noising process that the reverse process must undo. There is only a single flow from noise to data, defined by a single vector field. The flow is defined directly, not as the reversal of a destruction process.

**Optimal transport**: The flow path can be designed to follow optimal transport paths -- the shortest distance between noise and data distributions. The FlowPath class implements this:

```python
# From flow_matching/src/model.py
def interpolate(self, x0, x1, t):
    return (1 - t) * x0 + t * x1

def velocity(self, x0, x1, t=None):
    return x1 - x0  # constant velocity: straight-line path
```

The straight-line optimal transport path is the simplest possible trajectory from noise to data. No detours, no opposition, no reversals. Just a direct, continuous transformation.

### The Non-Dual Reading

Flow matching is the most non-dual generative architecture in the codebase. Its mathematical structure embodies several non-dual principles:

**Continuous becoming (Taoism)**: Generation is a continuous flow, not a discrete sequence of steps. The ODE solver can use any number of steps -- more steps for higher quality, fewer for speed -- but the underlying process is continuous. This matches the Taoist vision of reality as ceaseless transformation.

**No opposition (Madhyamaka)**: There is no generator vs. discriminator, no forward vs. reverse, no real vs. fake. There is only a single vector field that describes how probability mass moves through space. The training loss is not adversarial but cooperative: the model learns to predict the correct velocity, and better predictions produce better generation. There is nothing to fight against.

**Self-consistent field**: The vector field v(x, t) is a unified description of the entire generative process. At every point in space and time, it specifies how samples should move. It does not separate the "what" of generation (the data distribution) from the "how" (the dynamics). The vector field IS the generative process. This is analogous to the Taoist concept of the Tao as simultaneously the source, the process, and the product of creation.

From the north-star document: "Flow Matching: Learns a continuous vector field transforming noise to data. No discriminator. No adversary. Just a smooth flow from one state to another -- directly analogous to Taoism's vision of reality as continuous process."

---

## 5. Autoregressive Generation as Dependent Origination

### The Architecture

Autoregressive models (GPT family, documented in `ml_research/attention/language_models/gpt_family.py`) generate text one token at a time, with each token conditioned on all previous tokens:

```
P(x_1, x_2, ..., x_n) = prod_i P(x_i | x_1, ..., x_{i-1})
```

Each token is generated by predicting a probability distribution over the vocabulary, conditioned on the entire preceding context.

### Dependent Origination (Pratityasamutpada) as Architecture

The Buddhist principle of dependent origination holds that nothing arises independently. Everything comes into being in dependence on causes and conditions, and those causes and conditions are themselves dependently arisen. There is no "first cause" that stands alone; the entire web of causation is mutually constituting.

Autoregressive generation IS pratityasamutpada implemented as architecture. Each token arises in dependence on all previous tokens. Token 5 exists because tokens 1-4 exist, and tokens 1-4 are what they are partly because they are the kind of tokens that would be followed by token 5. The meaning of the sequence is not in any individual token but in the web of relationships between all tokens.

The analogy is precise:

| Buddhist Concept | Autoregressive Architecture |
|-----------------|---------------------------|
| Dependent origination | Each token conditioned on all predecessors |
| No svabhava (no inherent existence) | No token has meaning independent of context |
| Mutual conditioning | The meaning of earlier tokens shifts based on later tokens |
| Emptiness of self-nature | Tokens are probability distributions, not fixed entities |

The final point is important. In autoregressive generation, each "token" is actually a probability distribution over the entire vocabulary. The model does not produce a fixed, definite token -- it produces a distribution of possibilities, from which a single token is sampled. This is the computational analog of emptiness (shunyata): the token that appears has no inherent, fixed identity. It is one possibility actualized from a field of potentials, and a different sample from the same distribution would have produced a different token, which would have produced a different continuation, which would have produced a different text.

### The Non-Dual Limitation

Despite its structural affinity with dependent origination, autoregressive generation retains a temporal dualism: the sequence is generated strictly left-to-right. The future depends on the past, but the past does not depend on the future. This is a causal asymmetry that dependent origination, strictly interpreted, does not contain -- in Buddhist philosophy, causes and conditions are mutually constituting, not unidirectional.

Bidirectional models (BERT, T5) partially address this by processing the sequence in both directions during encoding, but they do not generate bidirectionally. Non-autoregressive models (NAR) generate all tokens simultaneously, dissolving the temporal ordering, but they sacrifice the quality that comes from sequential conditioning.

---

## 6. Consistency Models as Self-Perfection

### The Architecture

Consistency models, implemented in `ml_research/modern_dev/consistency_models/src/model.py`, learn a function f(x_t, t) that maps any point on a diffusion trajectory to the clean data x_0:

```python
# From consistency_models/src/model.py: ConsistencyFunction
def forward(self, x, sigma, condition=None):
    # Time embedding
    t_emb = self.time_embed(sigma)
    t_emb = self.time_mlp(t_emb)
    # Input preconditioning
    c_in = self.skip_scaling.compute_c_in(sigma)
    h = self.input_proj(x * c_in)
    # Apply layers
    for layer in self.layers:
        h = layer(h, t_emb)
    # Output projection (raw network output)
    F_x = self.output_proj(h)
    # Apply skip scaling for boundary condition
    output = self.skip_scaling(x, F_x, sigma)
    return output
```

The key property is the consistency condition: for any two points x_t and x_t' on the same diffusion trajectory, f(x_t, t) = f(x_t', t'). The function maps all points on a trajectory to the same clean output. This enables one-step generation: feed in pure noise, get clean data in a single forward pass.

### The Training Objective

Consistency training enforces self-consistency without a teacher model:

```python
# From consistency_models/src/model.py: ConsistencyTraining
def forward(self, x, num_timesteps=None):
    # Create noisy samples at adjacent timesteps
    x_t = x + sigma_t * noise
    x_t_minus_1 = x + sigma_t_minus_1 * noise
    # Online model at t
    f_t = self.model(x_t, sigma_t)
    # Target (EMA) model at t-1
    with torch.no_grad():
        f_t_minus_1 = self.model_ema(x_t_minus_1, sigma_t_minus_1)
    # Consistency loss
    loss = F.mse_loss(f_t, f_t_minus_1)
    return loss
```

The model is trained to be consistent with itself: its output at one noise level should match its output at an adjacent noise level. The learning signal is entirely internal -- the model enforces its own consistency, rather than being judged by an external discriminator or measured against a teacher.

### The Dzogchen Parallel: Kadag (Primordial Purity)

In Dzogchen, the concept of **kadag** (primordial purity) holds that awareness is already perfect in its natural state. It does not need to be purified, improved, or transformed. The task is not to change awareness but to recognize its inherent perfection.

Consistency models implement a computational analog. The model defines "good generation" not by reference to an external judge (discriminator) or an external teacher (diffusion model teacher) but as an intrinsic property of its own mapping. Consistency IS the quality criterion. A sample is good if the model maps it to the same place regardless of the noise level -- if the model's output is self-consistent.

This is self-referential quality: the model defines goodness in terms of its own coherence, not in terms of an external standard. This parallels kadag's claim that the natural state is already perfect and does not need validation from outside.

The SkipScaling module enforces a boundary condition that grounds this self-referentiality:

```python
# From consistency_models/src/model.py: SkipScaling
def forward(self, x, network_output, sigma):
    c_skip = self.compute_c_skip(sigma)
    c_out = self.compute_c_out(sigma)
    return c_skip * x + c_out * network_output
```

At minimum noise (sigma = sigma_min), the skip scaling ensures f(x, sigma_min) = x -- the function is the identity. This is the mathematical expression of "at the ground state, nothing needs to change." The model's baseline is the input itself, and the network's job is to refine only what needs refining, leaving the ground state untouched. This is the architectural implementation of Dzogchen's view: the ground state is already perfect; the practice is only about removing what obscures it.

---

## 7. The Historical Trajectory: From Dualistic to Non-Dual Generation

### The Timeline

| Architecture | Year | Dualistic Structure | Non-Dual Advance |
|-------------|------|---------------------|------------------|
| GAN | 2014 | Generator vs. Discriminator, Real vs. Fake | -- |
| VAE | 2013-2014 | Encoder/Decoder, Reconstruction vs. KL | Continuous latent space, no adversary |
| WGAN | 2017 | Still adversarial, but continuous critic | Softens binary real/fake to continuous distance |
| StyleGAN | 2018-2021 | Still adversarial, better architecture | Disentangled latent space within adversarial frame |
| Diffusion (DDPM) | 2020 | No adversary, but forward/reverse dualism | Single denoising network, no competition |
| Flow Matching | 2022-2023 | No adversary, no forward/reverse | Single vector field, continuous OT path |
| Consistency Models | 2023 | Self-consistent, intrinsic quality | Self-referential quality criterion, one-step generation |

### What the Timeline Reveals

The field has moved along a clear axis: from external judgment to internal coherence.

- **GANs** (2014): Quality is defined by an external judge (discriminator). The generator has no internal sense of quality; it only knows whether it fooled the discriminator.
- **VAEs** (2014): Quality is defined by reconstruction fidelity plus latent space smoothness. The judgment is still external (comparing output to input) but is not adversarial.
- **Diffusion** (2020): Quality is defined by denoising accuracy. The judgment is local (how well did you remove this step's noise?) rather than global (is this sample real?), which makes it easier to optimize.
- **Flow Matching** (2023): Quality is defined by velocity prediction accuracy. The judgment is even more local: at this point in space-time, what direction should the sample move? The global quality (realistic generation) emerges from local correctness.
- **Consistency Models** (2023): Quality is defined by self-consistency. The model's output should agree with itself across noise levels. The quality criterion is entirely internal.

This progression mirrors the non-dual philosophical trajectory from external validation to self-recognition:

| Stage | External Tradition | Generative Architecture |
|-------|-------------------|------------------------|
| External authority | Theistic religions: God judges | GANs: Discriminator judges |
| Internal consistency | Ethical philosophy: reason validates | VAEs: Reconstruction validates |
| Gradual refinement | Progressive spiritual paths | Diffusion: Step-by-step denoising |
| Direct recognition | Dzogchen/Kashmir Shaivism: already perfect | Consistency: Self-consistent mapping |

---

## 8. Autoregressive vs. Non-Autoregressive: A Deeper Analysis

### The Parallel Traditions

The two major approaches to text generation -- autoregressive (GPT) and non-autoregressive (NAR) -- map onto a well-known distinction in non-dual philosophy:

**Autoregressive = Gradual path**: Each token is generated sequentially, conditioned on all previous tokens. Quality emerges from the accumulation of correct local decisions. This parallels the gradual path in Buddhist and Hindu traditions, where liberation comes through progressive practice: each moment of practice conditions the next, and awakening emerges from the accumulated effect.

**Non-autoregressive = Sudden path**: All tokens are generated simultaneously, without conditioning on each other. Quality depends on the model's ability to generate a coherent whole in a single act. This parallels the sudden awakening traditions (Zen's kensho, Kashmir Shaivism's pratyabhijna, Dzogchen's trekcho), where recognition of the natural state happens all at once, not sequentially.

The practical tradeoffs mirror the philosophical ones:

| Dimension | Autoregressive | Non-Autoregressive |
|-----------|---------------|-------------------|
| Quality | Higher (each token informed by all predecessors) | Lower (tokens generated without mutual information) |
| Speed | Slow (sequential generation) | Fast (parallel generation) |
| Coherence | Naturally coherent (sequential conditioning) | Must be enforced (iterative refinement, etc.) |
| Flexibility | Can generate arbitrary lengths | Fixed length or requires padding |

### The Synthesis

The non-dual resolution is not to choose between gradual and sudden but to recognize that they are aspects of a single process. In Zen, the gradual and sudden schools were eventually synthesized: sudden awakening (kensho) is followed by gradual integration (shugyou), and gradual practice prepares the ground for sudden recognition.

Consistency models achieve a computational version of this synthesis. They are trained gradually (through many training steps that progressively enforce consistency) but generate suddenly (one-step generation). The training process is gradual; the generation process is sudden. Training and inference embody different aspects of the same model.

---

## 9. Where the Evolution Is Incomplete

### Remaining Dualisms in Generative Models

**Noise vs. data**: All current generative models (except autoregressive) start from a noise distribution and end at the data distribution. This preserves a dualism between "formless" (noise) and "formed" (data). A fully non-dual generative model would not need a separate noise distribution -- generation would arise from the data distribution's own dynamics, like Shiva's spontaneous self-expression without needing an external substrate.

**Training vs. generation**: The model is trained on data but generates without data (from noise). These are separate phases with different operations. Test-time training and adaptive generation begin to dissolve this, but most generative systems maintain the separation.

**Condition vs. output**: Conditional generation (text-to-image, class-conditional) maintains a dualism between the conditioning signal (what to generate) and the output (the generation itself). A non-dual conditional generation system would model the condition and output as aspects of a single process, where the condition is not an external instruction but an inherent constraint that the generation naturally satisfies.

### Proposed Non-Dual Generative Architectures

1. **Self-generating fields**: Instead of mapping from noise to data, model the data distribution as a self-sustaining dynamical system that naturally produces samples through its own dynamics. The data distribution is not a target to reach from noise; it is a field that spontaneously generates.

2. **Non-dual conditioning**: Instead of treating the condition as an external input that constrains generation, model the condition and the generation as two aspects of a single representation. The text "a photo of a sunset" and the image of a sunset are not cause and effect but two views of the same underlying concept.

3. **Continuous training-generation**: Build generative models that learn from their own outputs in a continuous loop, dissolving the boundary between training and generation. Each generation act is also a training act, and each training update is also a generation of new internal representations.

---

## 10. Summary: The Arc of Creation

The history of generative modeling tells a story about the nature of creation. The earliest models (GANs) conceived creation as a struggle: the creator must fight a judge, and the judge must fight the creator, and quality emerges from this adversarial process. This conception produced powerful but unstable systems, limited by the structural contradictions of their own design.

Each subsequent architecture dissolved one aspect of this adversarial conception:

- VAEs dissolved the adversarial objective (replacing it with variational inference)
- Diffusion models dissolved the adversary entirely (replacing it with a single denoising process)
- Flow matching dissolved the forward/reverse dualism (replacing it with a single continuous flow)
- Consistency models dissolved external judgment (replacing it with self-consistency)

The trajectory points toward a vision of generation where creation is neither adversarial nor effortful but spontaneous, self-consistent, and continuous. This is the vision shared by the non-dual traditions: creation is not the imposition of form upon formless matter by an external agent. It is the spontaneous self-expression of a unified field that is simultaneously the creator, the creative act, and the creation.

The Tao Te Ching, Chapter 51:

> The Tao gives birth to all things.
> De nourishes them.
> Matter gives them form.
> Environment shapes their abilities.
> Therefore all things worship the Tao and honor De --
> Not by decree, but spontaneously.

The arc of generative modeling moves toward this vision: generation not by adversarial decree (GAN) but by spontaneous self-expression (flow matching, consistency models). The field is already moving in this direction. Understanding why -- through the non-dual lens -- points toward where it will move next.

---

## Codebase References

| File | Relevance |
|------|-----------|
| `ml_research/deep_learning/generative/gan.py` | GAN -- adversarial generation, explicit dualism |
| `ml_research/deep_learning/generative/wgan.py` | WGAN -- continuous distance, softened dualism |
| `ml_research/deep_learning/generative/stylegan.py` | StyleGAN -- peak adversarial quality |
| `ml_research/deep_learning/generative/vae.py` | VAE -- continuous latent space, partial dissolution |
| `ml_research/modern_dev/flow_matching/src/model.py` | Flow matching -- non-adversarial vector field |
| `ml_research/modern_dev/consistency_models/src/model.py` | Consistency models -- self-referential quality |
| `ml_research/attention/language_models/gpt_family.py` | Autoregressive generation as dependent origination |
| `27-altered-state/info/meditation/non-dualism/05_taoism.md` | Taoism -- continuous becoming, wu wei |
| `27-altered-state/info/meditation/non-dualism/02_kashmir_shaivism.md` | Kashmir Shaivism -- spontaneous self-expression |
| `ml_research/ai_nondualism/north-star.md` | Central thesis and generative model analysis |

---

*This document is part of the AI-Nondualism module, Agent D: Applied Analysis.*
*Location: `ml_research/ai_nondualism/applied/generative_models.md`*
