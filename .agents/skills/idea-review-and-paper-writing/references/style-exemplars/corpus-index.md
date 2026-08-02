# User-published paper corpus index

The corpus was inspected from the local folder
`D:\【论文】\0... Accepted\论文❌9` on 2026-08-01. Page references in the exemplar
files use PDF page numbers, not printed journal page numbers.

## Corpus

| ID | Paper | Venue and year visible in source | Language | Observed high-level organization | Style use |
|---|---|---|---|---|---|
| [P01](../../cases/fault-diagnosis/published-edbn-fault-classification-style.md) | *A Novel Deep Learning Based Fault Diagnosis Approach for Chemical Process With Extended Deep Belief Network* | ISA Transactions, 2019 | English | Introduction; neural-network preliminaries; EDBN method; case studies; concluding remarks | Structural history; modernize local grammar before reuse |
| [P02](../../cases/fault-diagnosis/published-cg-sae-fault-classification-style.md) | *A Classification-Driven Neuron-Grouped SAE for Feature Representation and Its Application to Fault Classification in Chemical Processes* | Knowledge-Based Systems, 2021 | English | Introduction; SAE background; CG-SAE; case studies; concluding remarks | Structural history and mechanism–consequence patterns; language repair may be needed |
| [P03](../../cases/fault-diagnosis/published-lcp-fault-isolation-style.md) | *Layer-Wise Contribution-Filtered Propagation for Deep Learning-Based Fault Isolation* | International Journal of Robust and Nonlinear Control, 2022 | English | Introduction; preliminaries/problem formulation; LCP method and extensions; experimental verification; conclusions | Primary architecture and problem-to-contribution source |
| [P04](../../cases/data-completion/published-am-dae-imputation-style.md) | *Imputation of Missing Values in Time Series Using an Adaptive-Learned Median-Filled Deep Autoencoder* | IEEE Transactions on Cybernetics, accepted-version source dated 2022 | English | Introduction; supervised DAE baseline; AM-DAE method; case studies; concluding remarks | Primary motivation, mechanism, training-stage, and experiment source |
| [P05](../../cases/process-monitoring/published-vae-ilvm-monitoring-style.md) | *VAE-Based Interpretable Latent Variable Model for Process Monitoring* | IEEE Transactions on Neural Networks and Learning Systems, 2024 | English | Introduction; preliminaries/problem formulation; interpretable indicator design; threshold design; simulations; conclusion | Primary question-led Introduction, theory, and independent-second-task source |
| [P06](../../cases/fault-diagnosis/published-fae-gan-fault-estimation-style.md) | *A New Generative Adversarial Networks-Based Fault Diagnosis Framework: Learning a Mapping to Estimate Fault* | Neurocomputing, 2025 | English | Introduction; preliminaries/problem formulation; FD framework; independent FE extension; examples; conclusion | Primary purpose clause, mapping, offline/online workflow, and result source |
| [P07](../../cases/fault-diagnosis/published-tdn-decoupled-residual-style.md) | *Generation of Uncorrelated Residual Variables for Chemical Process Fault Diagnosis via Transfer Learning-Based Input–Output Decoupled Networks* | IEEE Transactions on Instrumentation and Measurement, 2025 | English | Introduction and Notations; preliminaries/problem formulation; TL-based IDN; experimental verification; conclusion | Primary macro-to-micro model description and notation source |
| [P08](../../cases/nonlinear-dynamics/published-memristive-multi-butterfly-style.md) | *A Memristive Neural Network With Hidden Multi-Butterfly Dynamics: Dynamics Analysis, Circuit Implementation, and Secure Encryption Application* | Physica Scripta, 2025 | English | Introduction; memristive HNN modeling; dynamic analysis; hardware; IoMT application; conclusion | Architecture for nonlinear-dynamics/application papers; not a default sentence bank |
| [P09](../../cases/fault-diagnosis/published-elm-aae-chinese-style.md) | *ELM-AAE驱动的工业过程故障诊断与故障深度估计* | 《控制理论与应用》网络首发, 2025 | Chinese | 引言；ELM-AAE；故障检测与估计；实验仿真；结论 | Primary Chinese cause–purpose–method sequence and terminology source |

## Selection principles

Use a passage as a positive exemplar only when it satisfies all of the following:

- its paragraph has one identifiable rhetorical purpose;
- the cause, limitation, or objective is connected to the response;
- the sentence form clarifies a genuine relation rather than decorating prose;
- the excerpt can be transferred without importing paper-specific truth;
- terminology is standard or can be verified in the active field;
- the local grammar is strong enough to imitate without silent repair.

When one of these conditions fails, retain the paper only as structural evidence
or turn the issue into a labeled negative example. Publication status alone does
not make every local sentence exemplary.

## Corpus-wide stable tendencies

Across the corpus, the strongest recurring pattern is:

`application need -> task definition -> method-family review -> shared limitation -> focused questions -> overall construction -> contributions -> roadmap`

The method sections then usually follow:

`overall mapping -> variables/interfaces -> component mechanism -> objective or condition -> training/algorithm -> diagnostic or predictive use`

The experiment sections usually follow:

`claim under test -> data and protocol -> metrics/baselines -> observed result -> mechanism-facing interpretation -> scope or limitation`

These tendencies support the skill's compact chapter sequence but do not require
every paper to use identical headings.
