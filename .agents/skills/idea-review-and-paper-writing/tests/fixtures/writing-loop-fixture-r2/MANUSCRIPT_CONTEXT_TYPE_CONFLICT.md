# Manuscript Context Type-Conflict Fixture

This negative fixture must fail for both an exact-symbol object-type conflict
and a project-specific symbol without a recorded rationale.

## Notation registry

| Symbol | Semantic family | Meaning | Naming basis / convention | Object type | Dimension | Typography | First definition | Scope |
|---|---|---|---|---|---|---|---|---|
| $x$ | state | Model state | field standard: state | scalar | $1$ | italic | Proposed Method before Eq. (1) | Global |
| $x$ | state | Model state | field standard: state | vector | $m_x$ | bold | Proposed Method before Eq. (2) | Global |
| $z$ | latent state | Project latent variable | project-specific: | scalar | $1$ | italic | Proposed Method before Eq. (3) | Global |
