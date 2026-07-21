# IEEE TFS reference patterns

The local `D:\_[PPPaper]\TFS_ref` library and its extracted roadmap support the following reusable patterns. They are guidance, not a source of unverifiable citations.

## Common model-to-proof chain

1. Define a nonlinear, T-S, IT2-T-S, Markov-jump, Lipschitz, or data-driven plant.
2. Specify actuator, sensor, intermittent, bias, degradation, communication, or unknown-input faults.
3. Construct an observer/filter/residual generator and write the augmented error system.
4. Establish Lyapunov, induced-norm, H-infinity, dissipative, interval, or zonotopic bounds.
5. Define residual statistics, adaptive/dynamic thresholds, isolation logic, or fault estimates.
6. Validate with a numerical benchmark and an application-oriented case, using explicit baselines and metrics.

## TFS-oriented narrative

- T-S/IT2 fuzzy representations are valuable because they preserve local linear structure for observer and stability analysis while representing nonlinear operating regimes.
- Robust TFS papers commonly make uncertainty explicit: membership uncertainty, partially known transition probabilities, asynchronous modes, sensor saturation, actuator nonlinearity, disturbances, and communication constraints.
- Fault-diagnosis papers distinguish detection from isolation and estimation. Sensor and actuator faults should not be collapsed into one physical channel without explaining the lifted or residual representation.
- Data-driven or deep fuzzy methods should explain what remains interpretable: premise variables, rules, memberships, local consequents, residuals, thresholds, or distilled labels.
- The experimental section should report false-alarm rate, missed-detection rate, delay, reconstruction RMSE/MAE, conditioning or residual-bound metrics, and ablation results when those claims are made.

## Residual-dynamics-based dynamic thresholds

- In the norm-based model-diagnosis tradition associated with Ding, a dynamic threshold is generated from the fault-free residual/error dynamics. It is not merely a moving statistical threshold fitted to recent residual samples.
- Decompose the nominal residual contribution into at least the propagated initial estimation error, the known/measured-input-dependent model mismatch or uncertainty, and disturbances/noise. Derive separate online recursions or reachable-set bounds before combining them into the detection threshold.
- For a time-varying T--S interpolation, condition the threshold generator on the current membership values and propagate the resulting time-varying matrices. Known inputs can tighten the bound because their instantaneous magnitude and direction are available online; bounded noise enters through a separate deterministic or probabilistic envelope.
- For a stacked residual, propagate the contributions over the entire window instead of treating the samples as independent blocks. Define the residual evaluation norm and show that the stacked fault-free residual remains within the generated bound.
- A confidence-qualified threshold may combine deterministic bounds for input/model uncertainty with a stated probabilistic bound for stochastic noise. State whether the guarantee is pointwise, window-wise, or sequential; do not attach a confidence level to an otherwise deterministic worst-case bound without a probability model.
- Prove fault detectability by comparing a lower bound on the stacked fault signature with the nominal dynamic threshold. A rank or minimum-singular-value condition prevents a nonzero fault direction from disappearing, while process-fault signatures that depend on the state require an excitation condition.
- Rolling empirical quantiles, moving covariance thresholds, and online residual recalibration are different methods. They may be used as auxiliary noise calibration, but they must not replace the residual-dynamics derivation when the paper claims a Ding-style dynamic threshold.

## Writing and citation discipline

- Organize the Introduction by problem tension and limitations, not by a year-by-year bibliography.
- Use references to support standard theory and prior methods; do not use the local extracted summaries as substitutes for verified BibTeX entries.
- Keep LMI, Lyapunov, H-infinity, event-triggered, interval-observer, zonotope, knowledge-distillation, and T-S terminology defined at first use.
- Do not copy equations or prose from the reference papers. Transfer structure and proof habits while deriving notation for the current manuscript.
