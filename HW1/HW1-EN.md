# Homework Assignment: IDS Evasion with Adversarial Attacks

## Overview

In this assignment, you will implement adversarial attacks against an Intrusion Detection System (IDS) trained on network traffic data. The goal is to understand both the capabilities and limitations of gradient-based evasion attacks, particularly when constrained by domain-specific semantic requirements.

**Learning Objectives:**

- Train and evaluate a binary classifier for intrusion detection
- Implement constrained adversarial attacks using Projected Gradient Descent (PGD)
- Understand the difference between mathematical evasion and semantically valid attacks
- Learn to integrate domain knowledge into adversarial attack generation

## The NSL-KDD Dataset

NSL-KDD is a refined version of the KDD Cup 1999 dataset, designed for intrusion detection research. It contains network connection records with 41 features spanning basic connection attributes (duration, protocol type, service, bytes transferred), content-based indicators (login attempts, compromised conditions), time-based traffic statistics (connection counts, error rates), and host-based network patterns (same-service rates, destination host statistics). Each connection is labeled as either normal traffic or one of various attack types (e.g., DoS, probe, R2L, U2R), including zero-day attacks in the test set that don't appear in the training data. The dataset contains a mix of categorical and numerical features

The dataset is publicly available and is also attached to the assignment in the `NSL-KDD` folder.

The file `nslkdd_features.json` contains detailed metadata for each feature, including:

- **Feature name and description**: Human-readable explanation of what the feature represents
- **Possible values**: The valid range or set of categorical values
- **Modifiability level**: Indicates which features an attacker can realistically modify during an evasion attempt without invalidating the attack itself

> **Important:** Not all features can be freely modified by an attacker. For example, features indicating successful compromise (e.g., `num_root`, `num_compromised`) cannot be changed by the attacker during evasion—modifying them would mean the attack has already succeeded or failed. The modifiability annotations in the JSON file guide which features should be perturbed during your adversarial attacks.

## Tasks

### Task 0: Data Preprocessing

Load and preprocess the NSL-KDD training and testing datasets. Your preprocessing pipeline should:

- **Merge feature information**: Combine training and testing data feature sets to ensure consistent encoding
- **One-hot encoding**: Convert the following categorical features to one-hot representation: `protocol_type`, `service`, `flag`
- **Standardization**: Apply StandardScaler to normalize all features (fit on training data, transform both training and testing)
- **Binary labels**: Convert multi-class attack labels to binary classification (0 = normal, 1 = attack)

**Note:** Save the fitted scaler and feature names—you'll need them for inverse transformations when checking attack plausibility.

**Important:** Do not perform any additional preprocessing beyond what is listed above. The model is designed to work with this specific preprocessing pipeline.

### Task 1: Train and Evaluate the IDS Binary Classifier

Implement and train the provided binary neural network architecture in PyTorch for intrusion detection. The model architecture is provided in `IDS.py`.

**Model Architecture:**
- Input layer: 122 features (after one-hot encoding)
- Hidden layer 1: 64 neurons, ReLU activation, Dropout(0.3)
- Hidden layer 2: 32 neurons, ReLU activation, Dropout(0.3)
- Output layer: 1 neuron, Sigmoid activation

**Training Hyperparameters:**
- Loss function: Binary Cross-Entropy (BCELoss)
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs: 10

After training, evaluate the model's performance on:

- **Complete test set**: Report overall accuracy, precision, recall, and F1-score
- **Zero-day attacks**: Attacks in the test set that were *not* present in the training data. Report accuracy on this subset.
  - *What are zero-day attacks?* These represent novel attack patterns that the IDS has never encountered during training. In real-world scenarios, zero-day attacks are particularly dangerous because traditional signature-based detection systems cannot recognize them. Evaluating on zero-day attacks tests the IDS's ability to generalize beyond known threats.
- **Non-zero-day attacks**: Attacks that *were* seen during training. Report accuracy on this subset.

**Remember:** When evaluating on test subsets, ensure you apply the same standardization (using the scaler fitted on training data) and convert to binary labels.

### Task 2: Evasion with Constrained PGD

Implement Projected Gradient Descent (PGD) attacks to evade the trained IDS, with the following important constraints and requirements:

**Attack setup:**

- Only attempt to evade test samples that are **correctly classified as attacks** by the IDS. (There's no point in attacking samples the IDS already misclassifies.)
- Goal: modify these attack samples to fool the IDS into classifying them as normal traffic

**Modifiability constraints:**

- Only modify features marked as highly/partially modifiable in `nslkdd_features.json`. Features like indicators of successful compromise cannot be changed without invalidating the attack.
- After each PGD iteration, truncate perturbed features to their valid ranges computed from the training data. Specifically, compute min/max for each feature from the training data, and use these to clamp perturbed samples.

**Epsilon values to test:** 0.05, 0.1, 0.15, 0.2, 0.3

**PGD Hyperparameters:**
- Number of iterations: 40
- Step size (alpha): 0.01
- Initialization: Start from the original sample (no random initialization for this assignment)

**For each epsilon value, report:**

- **Number of successful evasions**: How many correctly-classified attacks were fooled into being classified as normal
- **Ratio of plausible attacks**: Among the successful evasions, what fraction passes the plausibility checks?

**Plausibility checking:**

Use the provided `simple_rules.py` script to verify that adversarial examples don't violate critical domain rules. For instance:

- Long duration with zero bytes transferred is suspicious
- Very short duration with massive data transfer is unrealistic
- High error rates require sufficient connection counts

A *plausible attack* is one that successfully evades the IDS *and* passes all plausibility rules—meaning it represents a realistic, functioning attack that slips through detection.

**Why this matters:** In practice, an adversarial example that mathematically fools the classifier but violates network traffic physics (e.g., transferring 1GB in 0.001 seconds) is useless—it won't actually work as an attack. This task highlights the gap between mathematical adversarial examples and real-world evasion.

### Task 3: PGD with Plausibility-Aware Loss

Enhance your PGD attack by incorporating a plausibility term into the loss function. This approach integrates domain knowledge directly into the optimization, rather than just filtering results afterward.

#### Step 1: Train a Gaussian Naive Bayes (GNB) model

- Train GNB on the **training data** using the **original (multi-class) attack labels**, not binary labels
- GNB models each feature as an independent Gaussian distribution per class, giving us a simple probabilistic model of what "normal" attack traffic looks like for each attack type
- **Limitation:** This model can only provide likelihoods for attack types seen during training. Therefore, this enhanced PGD cannot be applied to zero-day attacks in the test set.

#### Step 2: Define the combined loss

Instead of just maximizing the probability of being classified as normal, we now optimize:

```
L_total = L_classifier + λ × L_plausibility
```

where:

- `L_classifier` is the binary cross-entropy loss with respect to the target (normal) class. Minimizing this tries to fool the IDS.
- `L_plausibility` is the _negative_ log-likelihood of the adversarial example under the GNB model, conditioned on the **original attack label**. Minimizing this keeps the example close to valid attack manifolds.
- `λ` (lambda) balances the two objectives. Use **λ = 0.1** for this assignment.

**Intuition:** The classifier loss pushes the example away from the attack manifold (toward the normal class). The plausibility loss pulls it back toward the attack manifold (to keep it realistic). The combined loss finds a sweet spot: close enough to the decision boundary to fool the IDS, but still resembling a valid attack.

#### Step 3: Compute the plausibility loss

For Gaussian Naive Bayes, the log-likelihood of a sample $\mathbf{x}$ given class $y$ is:
$$
\log p(\mathbf{x} | y) = \sum_i \log p(x_i | y)
$$
where each feature $i$ is modeled as:
$$
p(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)
$$
where $\mu_i$ and $\sigma_i$ are the mean and standard deviation of feature $i$ for class $y$ (extracted from the trained GNB model).

This gradient of $\log p(x_i | y)$  pulls $x$ toward the class mean $\mu$, encouraging the adversarial example to stay within the typical feature distribution of the original attack type.

#### Step 4: Implementation and evaluation

- Implement PGD with the combined loss function
- Run it on the same set of correctly-classified test attacks as in Task 2 (excluding zero-day attacks)
- Use the same epsilon values as Task 2
- For each epsilon, report:
  - Number of successful evasions
  - Ratio of plausible attacks (using the same plausibility checker as Task 2)

### Task 4: Analysis Questions

Answer the following questions in your submission:

1. Compare Task 2 and Task 3 results: does the plausibility-aware loss improve the ratio of plausible attacks? Why or why not?
   
2. Why is PGD with Plausibility-Aware Loss (Task 3) superior to the following alternatives?
    - _Alternative A - Post-hoc filtering:_ Running constrained PGD from Task 2 and then rejecting the generated adversarial examples that violate the plausibility checks afterward.
    - _Alternative B - Step-wise rejection:_ Performing plausibility checks after every gradient descent step during the attack and rejecting steps that would lead to implausible examples. 
      
## Submission Guidelines

Submit a Jupyter notebook or Python script containing:

- Complete implementation of all tasks
- Clear documentation and comments explaining your code
- Results tables showing:
  - IDS accuracy on test set, zero-day attacks, and non-zero-day attacks
  - For Task 2 and Task 3: number of successful evasions and plausibility ratios for each epsilon value
- Brief analysis (Task 4: 1-2 paragraphs) 
- Single submission per team is enough

