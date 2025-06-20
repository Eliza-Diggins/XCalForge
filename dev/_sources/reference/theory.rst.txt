Theory
======

xrmodifier is a framework for fitting temperature discrepancies between two X-ray telescopes by learning modifications to telescope calibration configurations, specifically the Ancillary Response File (ARF). The goal is to infer a global calibration correction that aligns posterior temperature estimates across instruments.

Contents:

1. Problem Overview_
2. Modification Model_
   - Gaussian ARF Modifier_
3. Computational Approach_
   - Step 1: Discretized Parameter Grid_
   - Step 2: Forward Simulation and Likelihood Computation_
     - a. Apply Calibration Modifier_
     - b. Simulate and Fit Spectra_
   - Step 3: Emulator Training_
     - Objective_
     - Model: Feedforward Neural Network (MLP)_

Problem Overview
----------------

Consider :math:`N` observations of astronomical sources with fixed but unknown temperatures :math:`T_i`, each observed by **two telescopes**:

- Telescope 0 (trusted): fixed configuration :math:`\Psi_0`
- Telescope 1 (biased): fixed configuration :math:`\Psi_1`

From these configurations and data, we obtain posterior distributions:

- :math:`P(T_{0,i} \mid \Psi_0)` — posterior from Telescope 0
- :math:`P(T_{1,i} \mid \Psi_1)` — posterior from Telescope 1

Assume :math:`\Psi_0` and :math:`\Psi_1` are constant across observations (i.e., same ARF, RMF, etc.).

We aim to find a **global modification** :math:`\phi \in \mathcal{F}` that adjusts Telescope 1’s configuration:

.. math::

    \Psi_1' = \phi(\Psi_1)

such that the modified Telescope 1 matches the Telescope 0 posteriors:

.. math::

    \Psi_1^* = \arg\min_{\phi \in \mathcal{F}} \sum_{i=1}^N \mathcal{D}\left[ P(T_{0,i} \mid \Psi_0), \, P(T_{1,i} \mid \phi(\Psi_1)) \right]

where :math:`\mathcal{D}` is a distance between distributions (e.g., KL divergence, Wasserstein, or squared difference in means).

If each telescope returns Gaussian posteriors or point estimates, this reduces to:

.. math::

    \mathcal{L}(\theta) = \sum_{i=1}^N \left[ T_{0,i} - T_{1,i}'(\theta) \right]^2

where:

- :math:`T_{0,i}`: trusted temperature from Telescope 0
- :math:`T_{1,i}'(\theta)`: modified temperature from Telescope 1
- :math:`\theta`: parameters of the calibration modification

Modification Model
------------------

We assume a parametric family :math:`\mathcal{F}` of calibration modifiers.

Gaussian ARF Modifier
^^^^^^^^^^^^^^^^^^^^^

.. math::

    \phi_\theta(\Psi_1)(E) = \Psi_1(E) \cdot \left[ 1 + A \cdot \exp\left(-\frac{(E - \mu)^2}{2\sigma^2} \right) \right]

- :math:`\theta = (\mu, \sigma, A)` are learned by optimization.

Computational Approach
----------------------

We build a surrogate model to efficiently evaluate calibration corrections.

Step 1: Discretized Parameter Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let :math:`\theta = (\alpha_1, \alpha_2, \ldots, \alpha_d)` parametrize the calibration modifier.

Discretize each parameter into :math:`N_i` values. Let:

.. math::

    \mathcal{T} = \{T_1, T_2, \ldots, T_{N_T} \}

Then the simulation grid is:

.. math::

    \mathcal{G} \times \mathcal{T} = \{ (\theta, T_{\text{true}}) \}

with total size:

.. math::

    |\mathcal{G} \times \mathcal{T}| = \left( \prod_{i=1}^d N_i \right) \cdot N_T

Step 2: Forward Simulation and Likelihood Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We simulate:

.. math::

    \mathcal{S}: \mathcal{F} \supset \Xi \mapsto T_{\rm mod}(T_{\rm unmod}|T_{\rm true})

An XSPEC model :math:`M` is used as a hyperparameter.

a. Apply Calibration Modifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute:

.. math::

    \Psi_1' = \phi_\theta(\Psi_1)

This is a new ARF calibration file used for XSPEC.

b. Simulate and Fit Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate synthetic spectrum at :math:`T_{\text{true}}`, then fit using both original and modified calibration:

Original fit:

.. math::

    T_1 = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1)

Modified fit:

.. math::

    T_1' = \arg\max_T \mathcal{L}(D_\theta \mid T, \Psi_1')

Store :math:`T_1`, :math:`T_1'`, and likelihood curves.

Step 3: Emulator Training
^^^^^^^^^^^^^^^^^^^^^^^^^

We train a neural network to emulate the calibration effect.

Objective
~~~~~~~~~

Learn a function:

.. math::

    f_{\text{emulator}}(\theta, T_{\text{true}}) \approx T_1'(\theta; T_{\text{true}})

where:

- :math:`\theta \in \mathbb{R}^d`: modifier parameters
- :math:`T_{\text{true}} \in \mathbb{R}`: true input temperature
- :math:`T_1'`: modified fitted temperature

Model: Feedforward Neural Network (MLP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Input**: :math:`(\theta, T_{\text{true}}) \in \mathbb{R}^{d+1}`
- **Hidden**: 2–3 fully connected layers (64–128 units, ReLU or GELU)
- **Output**: scalar prediction :math:`T_1'`
- **Loss**: Mean Squared Error (MSE)
