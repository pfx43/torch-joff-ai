# Typical manuscript errors and corrections

Use this catalog as a negative checklist. The examples are generalized from recurring author feedback; they are not paper-specific notation requirements.

## Terminology and manuscript voice

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| `healthy transition` | Literal and nonstandard control terminology; the adjective does not identify whether the model, state equation, trajectory, or data regime is meant. | Use `nominal state-transition mapping`, `fault-free state equation`, `normal-operation model`, or `normal trajectory` according to the object. |
| “The paper-specific ... is introduced in Section III; it is not part of these preliminaries.” | Drafting commentary tells the reader how the manuscript was assembled instead of advancing the technical argument. | End the preliminaries after the standard definition; introduce the construction directly where it is defined. |
| “Importantly, the learned state need not ...” when the sentence is not used in an assumption, result, or derivation | Authorial aside or unsupported interpretation that often reads as generated filler. | State the precise mathematical requirement where needed, or delete the sentence. |
| “The current manuscript package does not contain ...” or “the completed submission must ...” | Describes the drafting process rather than the scientific method. | Present a validation protocol in impersonal scientific language, and label genuinely missing evidence outside the manuscript body. |

## Symbol families and fonts

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Reusing `F` for a state-transition mapping while `f` denotes faults | Uppercase/lowercase/font variants still belong to one visual base family and suggest a common meaning. | Reserve the `f/F` family for fault quantities and choose a different calligraphic mapping family for state evolution. |
| Using `n` both for normal/nominal status and for a dimension, such as `n_x` | One base symbol carries unrelated semantics. | Reserve `n` for normal/nominal quantities and use a dimension prefix such as `m_x`. |
| Using `w` for noise while `W`, `\omega`, or `\Omega` denotes weights or whitening objects | The same visual family mixes disturbances with weights. | Use a dedicated disturbance family such as `\varepsilon`; retain `W/\omega/\Omega` for weights or whitening. |
| Writing a measured output as `y^m` when no competing output family requires the superscript | Adds visual clutter and can be misread as an exponent or an independent semantic variant. | Use `y` for the measured output and decorate only genuinely different predicted or nominal outputs. |
| Using a blackboard-bold letter such as `\mathbb I` for a stacked history or information variable | Blackboard bold conventionally denotes number fields, sets, domains, or spaces, not an ordinary data vector. | Use a bold vector or matrix and display its explicit stacked form. |

## Equation construction

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| `\operatorname{col}(a,b,c)` for a vector or matrix stack | Reads like a programming helper and hides orientation and block dimensions. | Write an explicit bracketed array, for example `[a^{\mathsf T}\ b^{\mathsf T}\ c^{\mathsf T}]^{\mathsf T}`. |
| A controlled Koopman operator equation whose left-hand side omits control or exogenous arguments shown on the right | Obscures whether the operator is autonomous, input-parameterized, or defined on an extended state. | Show the control/exogenous arguments consistently in the operator action, or define an extended state and shift map explicitly. |
| Calling a repeated operator/commutative identity a “finite-dimensional Koopman model” | The second equation adds no approximation or finite-dimensional dynamics. | Introduce a finite lifting and write a distinct controlled linear predictor, e.g. `z_{k+1}=Az_k+B_u u_k+B_\xi \xi_k`, with a reconstruction/output relation and citations. |
| Using two displayed equations that express the same identity with only scalar/vector notation changed | Inflates equation count without adding a new result. | Combine them or make the later equation perform the next mathematical step. |

## Section organization and focus

| Incorrect pattern | Why it fails | Preferred correction |
|---|---|---|
| Ending Section II with `Monitoring Objectives` followed by a long prose checklist | Does not formulate the paper's mathematical problem and fragments the contribution into implementation details. | End Section II with `Problem Formulation` and state two central mathematical objectives; use at most three only when indispensable. |
| More than three primary objectives in the problem formulation | Usually indicates that the paper lacks a focused technical question. | Merge dependent requirements under two central tasks, such as detection and structured isolation. |
