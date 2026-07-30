# Typical manuscript errors and corrections

Use this catalog as a negative checklist. The examples are generalized from recurring author feedback; they are not paper-specific notation requirements.

## Terminology and manuscript voice

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| `healthy transition` | Literal and nonstandard control terminology; the adjective does not identify whether the model, state equation, trajectory, or data regime is meant. | Use `nominal state-transition mapping`, `fault-free state equation`, `normal-operation model`, or `normal trajectory` according to the object. |
| “The paper-specific ... is introduced in Section III; it is not part of these preliminaries.” | Drafting commentary tells the reader how the manuscript was assembled instead of advancing the technical argument. | End the preliminaries after the standard definition; introduce the construction directly where it is defined. |
| “Importantly, the learned state need not ...” when the sentence is not used in an assumption, result, or derivation | Authorial aside or unsupported interpretation that often reads as generated filler. | State the precise mathematical requirement where needed, or delete the sentence. |
| “The current manuscript package does not contain ...” or “the completed submission must ...” | Describes the drafting process rather than the scientific method. | Present a validation protocol in impersonal scientific language, and label genuinely missing evidence outside the manuscript body. |
| “For the first time,” “the first,” “pioneering,” or “fills the gap” without exhaustive verification | Makes an absolute priority claim that normal literature review cannot establish reliably. | State the exact construction, theorem, algorithm, or application introduced by the paper. |
| Calling a standard attention block or a direct combination of two known methods the principal innovation | Names ingredients without identifying a new technical consequence. | State the new model property, computable condition, algorithm, guarantee, or validated capability. |

## Symbol families and fonts

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Reusing `F` for a state-transition mapping while `f` denotes faults | Uppercase/lowercase/font variants still belong to one visual base family and suggest a common meaning. | Reserve the `f/F` family for fault quantities and choose a different calligraphic mapping family for state evolution. |
| Using `n` both for normal/nominal status and for a dimension, such as `n_x` | One base symbol carries unrelated semantics. | Reserve `n` for normal/nominal quantities and use a dimension prefix such as `m_x`. |
| Using `w` for noise while `W`, `\omega`, or `\Omega` denotes weights or whitening objects | The same visual family mixes disturbances with weights. | Use a dedicated disturbance family such as `\varepsilon`; retain `W/\omega/\Omega` for weights or whitening. |
| Writing a measured output as `y^m` when no competing output family requires the superscript | Adds visual clutter and can be misread as an exponent or an independent semantic variant. | Use `y` for the measured output and decorate only genuinely different predicted or nominal outputs. |
| Using a blackboard-bold letter such as `\mathbb I` for a stacked history or information variable | Blackboard bold conventionally denotes number fields, sets, domains, or spaces, not an ordinary data vector. | Use a bold vector or matrix and display its explicit stacked form. |
| Identifying a Koopman, GRU, or LSTM latent state with the physical state without a coordinate assumption | Makes physical measurements and state-space arguments appear available when they are not. | Treat it as a learned latent state and define any injectivity, reconstruction, or output-map assumption explicitly. |

## Equation construction

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| `\operatorname{col}(a,b,c)` for a vector or matrix stack | Reads like a programming helper and hides orientation and block dimensions. | Write an explicit bracketed array, for example `[a^{\mathsf T}\ b^{\mathsf T}\ c^{\mathsf T}]^{\mathsf T}`. |
| A controlled Koopman operator equation whose left-hand side omits control or exogenous arguments shown on the right | Obscures whether the operator is autonomous, input-parameterized, or defined on an extended state. | Show the control/exogenous arguments consistently in the operator action, or define an extended state and shift map explicitly. |
| Calling a repeated operator/commutative identity a “finite-dimensional Koopman model” | The second equation adds no approximation or finite-dimensional dynamics. | Introduce a finite lifting and write a distinct controlled linear predictor, e.g. `z_{k+1}=Az_k+B_u u_k+B_\xi \xi_k`, with a reconstruction/output relation and citations. |
| Using two displayed equations that express the same identity with only scalar/vector notation changed | Inflates equation count without adding a new result. | Combine them or make the later equation perform the next mathematical step. |
| Reusing a physical output matrix directly on a learned lifted state | Assumes an unproved physical-to-latent coordinate relation and may be dimensionally wrong. | Define a separate learned output matrix or nonlinear decoder and state the required reconstruction assumptions. |

## Theory, information, and implementation

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Proving a linearized error recursion and claiming global nonlinear stability | Ignores nonlinear remainders, Jacobian variation, and the region in which the approximation holds. | State a local or first-order result, or bound the nonlinear remainder and establish a domain of attraction with a uniform margin. |
| Treating a plant Jacobian, true hidden state, unknown fault direction, or counterfactual fault-free trajectory as an online design variable | Uses information that the stated problem does not provide. | Classify every quantity as measured, known, learned, estimated, bounded, designable, or unavailable and derive the method using only admitted information. |
| Fixing clusters or fuzzy prototypes in a latent space while continuing to train the encoder | The coordinate system changes, so the supposedly fixed prototypes may no longer represent the learned data. | Jointly or alternately update the representation and prototypes, or justify why the fixed representation remains valid. |
| Calling a soft regularization loss a certified hard constraint | A finite penalty does not guarantee satisfaction of the claimed inequality. | Use a structural parameterization or projection, or describe the condition honestly as a regularizer and verify it after training. |
| Claiming exact spectral normalization when computing only a Frobenius bound | The Frobenius norm is an upper bound, not the exact spectral norm. | Name the implemented Frobenius constraint and prove only the induced guarantee it supports. |
| Requiring online clustering, SDP, SVD, large inverses, or long-window high-dimensional optimization without a deployment analysis | Leaves the claimed online method computationally undefined or impractical. | Move design offline, use fixed or recursive low-dimensional structures, and report online per-sample complexity. |
| Describing attention weights as fuzzy memberships without a membership construction | Feature importance and rule-membership degree have different semantics and constraints. | Define premise features, a nonnegative normalized membership mapping, and how its parameters are learned. |
| Using current potentially faulty measurements to update both the nominal reference and the residual without analyzing leakage | The fault may contaminate the reference and cancel its own residual signature. | Use a justified measurement-decoupled reference route or model and bound the measurement-to-reference leakage explicitly. |
| Claiming that a joint residual is better because it contains “more information” | Quantity of signals does not establish detectability. | Prove a null-space, rank, conditioning, detectable-subspace, or minimum-gain improvement. |
| Interpreting a standard basis vector in a learned lifted space as a physical component fault | Learned coordinate directions do not automatically correspond to sensors, actuators, or components. | Call it a candidate lifted disturbance channel unless a physical mapping and evidence are supplied. |

## Experiments and evidence

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Writing “experiments demonstrate” when only an experiment plan exists | Converts missing evidence into a fabricated result. | Describe the protocol, metric, and intended validation object without reporting an outcome. |
| Claiming normal-only training while using fault labels to select thresholds, checkpoints, or hyperparameters | Violates the declared information boundary and introduces label leakage. | Keep fault labels out of model selection or revise the training assumption explicitly. |
| Reporting only average accuracy for a robustness, stability, detectability, or dynamic-threshold claim | The metric does not test the property claimed by the theory. | Add false alarms, delay, coverage, minimum-gain cases, error decay, conditioning, robustness, or constraint verification as appropriate. |

## Scientific figures

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Using an AI-generated concept image as the only final source | Text, arrows, topology, and scientific details may be wrong, while the flattened image cannot be revised reliably. | Audit the concept against the manuscript and reconstruct accepted content from editable primitives. |
| Reproducing the distinctive layout of an unlicensed published figure because the drawing tool is open source | Tool licensing does not grant rights to the reference figure's expressive design. | Transfer the scientific relationships into an independently organized composition and retain provenance. |
| Mixing several attractive palette strips or using the accent on ordinary modules | Breaks semantic consistency and removes the visual hierarchy the accent was meant to create. | Select one strip, keep cool or neutral tones dominant, and reserve one bright swatch for a defined meaning. |
| Embedding a bitmap in PowerPoint and calling every element editable | The container is editable, but the scientific elements inside the image are not. | Build native PowerPoint shapes, text, connectors, and groups, or deliver another honestly labeled editable vector source. |

## Section organization and focus

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Ending Section II with `Monitoring Objectives` followed by a long prose checklist | Does not formulate the paper's mathematical problem and fragments the contribution into implementation details. | End Section II with `Problem Formulation` and state two central mathematical objectives; use at most three only when indispensable. |
| More than three primary objectives in the problem formulation | Usually indicates that the paper lacks a focused technical question. | Merge dependent requirements under two central tasks, such as detection and structured isolation. |
| Writing sections before assigning their single question and output | Encourages duplicated responsibilities, misplaced theory, and a table of contents that follows drafting history rather than scientific dependency. | Complete the section-role matrix first and block drafting until every planned section has one primary question and one output. |
| Adding symbols locally and planning to reconcile notation at the end | Allows base-character collisions and inconsistent fonts to spread through equations, proofs, algorithms, and figures. | Register each object before use and rerun the notation check after every substantive subsection edit. |
| An abstract, problem statement, and contribution list with different task counts or order | Signals that the paper is presenting several incompatible main lines. | Use the same two or three subproblems in the same order and map each to one body result. |
| Allowing the table of contents to drift as modules are added | Produces a structure based on drafting history instead of the skill's scientific sequence. | Recheck the complete chapter arrangement in every loop and record a target-journal or scientific justification for each deviation. |
| Grammatically correct sentences placed next to one another without a logical dependency | Creates fluent-looking prose whose claims, referents, and transitions cannot be audited. | Assign each sentence a role and make its relation to the preceding and following argument explicit. |
| Listing modules in execution order and treating that order as causal explanation | Chronology does not explain why a module is needed or how it produces the claimed result. | Trace problem or limitation to method need, mathematical action, consequence, and validating evidence. |
| Giving equations without dimensions, conditions, intermediate steps, or a scoped conclusion | Prevents verification and often turns a local or conditional relation into an overstated result. | Audit definitions, assumptions, dimensions, indices, algebraic transitions, boundary cases, and conclusion scope after every relevant subsection. |
| Describing a model only by architecture names or a block diagram | Omits its information boundary, governing mappings, update order, and component interfaces. | State inputs, outputs, states, known and unknown quantities, fixed and learned mappings, parameters, assumptions, initialization, and update sequence. |
| Mixing training, validation, testing, and online inference into one workflow | Hides model-selection rules and permits label, preprocessing, future-time, threshold, or test-set leakage. | Specify separate reproducible flows, fitted preprocessing, allowed labels, objectives, checkpoints, test-only operations, metrics, and offline/online boundaries. |
