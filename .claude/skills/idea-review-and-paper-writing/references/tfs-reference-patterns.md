# IEEE TFS reference patterns

When a local TFS reference library or extracted roadmap is supplied for the task, use it to support the following reusable patterns. It is guidance, not a substitute for verified primary sources or BibTeX records.

This document contains reusable IEEE TFS positioning and proof patterns. A special discussion about one TFS manuscript belongs in its matching file under `cases/<task-group>/`, not here. Promote only the generalized lesson from that case into this document.

## Common model-to-proof chain

1. Define a nonlinear, T-S, IT2-T-S, Markov-jump, Lipschitz, or data-driven plant.

2. Specify actuator, sensor, intermittent, bias, degradation, communication, or unknown-input faults.

3. Construct an observer/filter/residual generator and write the augmented error system.

4. Establish Lyapunov, induced-norm, H-infinity, dissipative, interval, or zonotopic bounds.

5. Define residual statistics, adaptive/dynamic thresholds, isolation logic, or fault estimates.

6. Validate with a numerical benchmark and an application-oriented case, using explicit baselines and metrics.

## TFS-oriented narrative

- When relevant, build the Introduction's technical tension from nonlinear operating regimes and uncertainty, through the limits of linear or unstructured model-free diagnosis, to the value of a structured T–S/IT2 representation, then state the exact unresolved issue and the paper's contribution.
- T-S/IT2 fuzzy representations are valuable because they preserve local linear structure for observer and stability analysis while representing nonlinear operating regimes.
- Robust TFS papers commonly make uncertainty explicit: membership uncertainty, partially known transition probabilities, asynchronous modes, sensor saturation, actuator nonlinearity, disturbances, and communication constraints.
- Fault-diagnosis papers distinguish detection from isolation and estimation. Sensor and actuator faults should not be collapsed into one physical channel without explaining the lifted or residual representation.
- Data-driven or deep fuzzy methods should explain what remains interpretable: premise variables, rules, memberships, local consequents, residuals, thresholds, or distilled labels.
- In a TFS-targeted paper, the fuzzy system should contribute materially to modeling, premise scheduling, error propagation, observer or residual construction, and robust analysis where those claims are made. Do not attach a T–S block only at the network output and treat its presence as the fuzzy-systems contribution.
- The experimental section should report false-alarm rate, missed-detection rate, delay, reconstruction RMSE/MAE, conditioning or residual-bound metrics, and ablation results when those claims are made.

## Residual-dynamics-based dynamic thresholds

Use the authoritative dynamic-threshold rules in `references/domains/fault-diagnosis.md`. For a TFS paper, make the membership-dependent time variation and the fuzzy system's role in the residual propagation explicit; do not create a second threshold definition here.

## Writing and citation discipline

- Organize the Introduction by problem tension and limitations, not by a year-by-year bibliography.
- Use references to support standard theory and prior methods; do not use the local extracted summaries as substitutes for verified BibTeX entries.
- Keep LMI, Lyapunov, H-infinity, event-triggered, interval-observer, zonotope, knowledge-distillation, and T-S terminology defined at first use.
- Do not copy equations or prose from the reference papers. Transfer structure and proof habits while deriving notation for the current manuscript.
