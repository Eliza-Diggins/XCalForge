# xrmodifier

**xrmodifier** is a framework for fitting temperature discrepancies between two X-ray telescopes by learning modifications to telescope calibration configurations, specifically the Ancillary Response File (ARF). The goal is to infer a global calibration correction that aligns posterior temperature estimates across instruments.

1. [🧭 Problem Overview](#-problem-overview)  
2. [🔧 Modification Model](#-modification-model)  
   - [Gaussian ARF Modifier](#gaussian-arf-modifier)  
3. [🧮 Computational Approach](#computational-approach)  
   - [Step 1: Discretized Parameter Grid](#step-1-discretized-parameter-grid)  
   - [Step 2: Forward Simulation and Likelihood Computation](#step-2-forward-simulation-and-likelihood-computation)  
     - [a. Apply Calibration Modifier](#a-apply-calibration-modifier)  
     - [b. Simulate and Fit Spectra](#b-simulate-and-fit-spectra)  
   - [Step 3: Emulator Training](#step-3-emulator-training)  
     - [📌 Objective](#-objective)  
     - [🧠 Model: Feedforward Neural Network (MLP)](#-model-feedforward-neural-network-mlp)  
       - [🛠 Architecture](#-architecture)  

---

## 🧭 Problem Overview

Consider $N$ observations of astronomical sources with fixed but unknown temperatures $T_i$, each observed by **two telescopes**:

- Telescope 0 (trusted): fixed configuration $\Psi_0$
- Telescope 1 (biased): fixed configuration $\Psi_1$

From these configurations and data, we obtain posterior distributions:

- $P(T_{0,i} \mid \Psi_0)$ — posterior from Telescope 0  
- $P(T_{1,i} \mid \Psi_1)$ — posterior from Telescope 1

Let us assume $\Psi_0$ and $\Psi_1$ are constant across observations (i.e., same ARF, RMF, etc. for each telescope across the dataset).


We aim to find a **global modification** $\phi \in \mathcal{F}$ that adjusts Telescope 1’s configuration (typically its ARF):

$$
\Psi_1' = \phi(\Psi_1)
$$

such that the posterior temperature estimates from the modified Telescope 1 match those from Telescope 0:

$$
\Psi_1^* = \arg\min_{\phi \in \mathcal{F}} \sum_{i=1}^N \mathcal{D}\left[ P(T_{0,i} \mid \Psi_0), \, P(T_{1,i} \mid \phi(\Psi_1)) \right]
$$

where $$\mathcal{D}$$ is a divergence or distance metric between distributions (e.g., KL divergence, Wasserstein distance, or squared difference in means).

If each telescope returns a Gaussian posterior (mean and standard deviation), or we work with point estimates (e.g., MAP or MLE temperatures), the objective reduces to minimizing:

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \left[ T_{0,i} - T_{1,i}'(\theta) \right]^2
$$

where:

- $T_{0,i}$: inferred temperature from Telescope 0 (trusted)
- $T_{1,i}'(\theta)$: inferred temperature from Telescope 1 using modified configuration $\phi_\theta(\Psi_1)$
- $\theta$: parameters of the calibration modification (e.g., for a Gaussian modifier to the ARF: center $\mu$, width $\sigma$, amplitude $A$)

## 🔧 Modification Model

We assume a parametric family $\mathcal{F}$ of calibration modifiers. Currently supported:

### Gaussian ARF Modifier

$$
\phi_\theta(\Psi_1)(E) = \Psi_1(E) \cdot \left[ 1 + A \cdot \exp\left(-\frac{(E - \mu)^2}{2\sigma^2} \right) \right]
$$

- $\theta = (\mu, \sigma, A)$ are learned via optimization.

# Computational Approach

For any given parameterized family of calibration modifications $\Xi \subset \mathcal{F}$, the computational workflow constructs a simulation-driven surrogate model that enables Bayesian inference of calibration corrections from observed temperature discrepancies.

### Step 1: Discretized Parameter Grid

Let $\Xi$ be parameterized by $d$ free parameters:

$$
\theta = (\alpha_1, \alpha_2, \ldots, \alpha_d)
$$

Each parameter is discretized into $N_i$ values, forming a grid:

$$
\theta \in \mathcal{G} = \{ \alpha_1^{(j_1)}, \ldots, \alpha_d^{(j_d)} \}, \quad j_i = 1, \ldots, N_i
$$

Let $\mathcal{T} = \{T_1, T_2, \ldots, T_{N_T} \}$ be a discretized set of true temperatures spanning the desired astrophysical range (e.g., from $T_{\min}$ to $T_{\max}$).

The full simulation grid is the Cartesian product:

$$
\mathcal{G} \times \mathcal{T} = \{ (\theta, T_{\text{true}}) \}
$$

with total grid size:

$$
|\mathcal{G} \times \mathcal{T}| = \left( \prod_{i=1}^d N_i \right) \cdot N_T
$$

### Step 2: Forward Simulation and Likelihood Computation

The goal of the forward modeling step is to produce a discretization of the map 

$$
\mathcal{S}: \mathcal{F} \supset \Xi \mapsto T_{\rm mod}(T_{\rm unmod}|T_{\rm true})
$$

To do this an XSPEC model $M$ is selected as a **hyperparameter**.

#### a. Apply Calibration Modifier

Compute the modified calibration:

$$
\Psi_1' = \phi_\theta(\Psi_1)
$$

where $\phi_\theta$ is a member of the modification family (e.g., a Gaussian multiplier applied to the ARF). This produces a new calibration file suitable for XSPEC.

#### b. Simulate and Fit Spectra

Using XSPEC, simulate a synthetic spectrum for temperature $T_{\text{true}}$ and model $M(T)$ under the modified calibration $\Psi_1'$.

Then, fit this simulated spectrum using both the original and modified calibrations:

- Fit under original calibration:
  
  $$T_1 = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1)
  $$

- Fit under modified calibration:
  
  $$T_1' = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1')
  $$

Store $T_1$, $T_1'$, and the associated likelihood curves for each $(\theta, T_{\text{true}})$.

---

### Step 3: Emulator Training

To enable efficient inversion of the temperature correction problem, we train a neural network to emulate the effect of calibration modifications on inferred temperatures.

#### 📌 Objective

We construct a surrogate model that approximates the simulation map:

$$
f_{\text{emulator}}(\theta, T_{\text{true}}) \approx T_1'(\theta; T_{\text{true}})
$$

where:

- $\theta \in \mathbb{R}^d$: calibration modifier parameters (e.g., $\mu$, $\sigma$, $A$ for a Gaussian ARF perturbation)
- $T_{\text{true}} \in \mathbb{R}$: the true (input) plasma temperature
- $T_1' \in \mathbb{R}$: the best-fit temperature recovered by Telescope 1 after applying modifier $\phi_\theta$

This emulator replaces the need to re-run XSPEC simulations during inverse optimization, making the inference step fast and differentiable.

---

#### 🧠 Model: Feedforward Neural Network (MLP)

We use a multilayer perceptron (MLP) to learn the mapping from $(\theta, T_{\text{true}}) \mapsto T_1'$.

##### 🛠 Architecture

- **Input**: concatenated vector $(\theta, T_{\text{true}}) \in \mathbb{R}^{d+1}$
- **Hidden Layers**: 2–3 fully connected layers (e.g., 64–128 units each) with ReLU or GELU activations
- **Output**: scalar prediction of the best-fit $\theta^\star$.
- **Loss Function**: Mean Squared Error (MSE)

