# 2D Potential Flow Solver (CFD) — Rust Implementation

## Overview

This project implements a 2D Computational Fluid Dynamics (CFD) solver for incompressible, irrotational flow using the **Laplace equation for velocity potential**:

[
\nabla^2 \phi = 0
]

The solver is based on the **finite difference method (FDM)** applied to a structured, non-uniform Cartesian mesh. It is designed with a modular architecture that allows different iterative schemes to be easily implemented and compared.

---

## Features

* Structured Cartesian mesh with geometric stretching
* Finite difference discretization on non-uniform grids
* Multiple iterative solvers:

  * Jacobi
  * Gauss-Seidel
  * Successive Over-Relaxation (SOR)
  * Line Gauss-Seidel (in progress / optional)
* Residual monitoring and convergence tracking
* Post-processing:

  * Velocity field computation
  * Pressure coefficient (Cp)
  * Airfoil Cp distribution
* CSV export for visualization (Paraview, Tecplot, Python, etc.)

---

## Governing Equations

The solver computes the velocity potential ( \phi ), from which velocity is obtained:

[
u = \frac{\partial \phi}{\partial x}, \quad v = \frac{\partial \phi}{\partial y}
]

The pressure coefficient is computed using Bernoulli’s equation:

[
C_p = 1 - \frac{u^2 + v^2}{U_\infty^2}
]

---

## Project Structure

```text
src/
│
├── main.rs                # Entry point (simulation pipeline)
├── config.rs              # Simulation parameters
├── mesh.rs                # Mesh generation
├── solver_core.rs         # Core solver loop
├── solver_utils.rs        # Post-processing and utilities
│
└── tm_schemes/            # Time-marching (iterative) schemes
    ├── mod.rs
    ├── jacobi.rs
    ├── gauss_seidel.rs
    ├── sor.rs
    └── line_gauss_seidel.rs
```

---

## Numerical Methods

All iterative methods implement the same interface:

```rust
trait TimeMarchingScheme {
    fn step(&self, mesh: &Array2<Node>, phi_n: &mut Array2<f64>) -> f64;
}
```

### Implemented Schemes

| Method       | Type       | Characteristics                       |
| ------------ | ---------- | ------------------------------------- |
| Jacobi       | Explicit   | Simple, slow convergence              |
| Gauss-Seidel | In-place   | Faster than Jacobi                    |
| SOR          | Relaxed GS | Accelerated convergence (ω-dependent) |

### Key Idea

All methods solve:

[
L(\phi) = 0
]

through iterative correction:

[
\phi^{n+1} = \phi^n - \frac{L(\phi^n)}{N}
]

where ( N ) is a diagonal approximation of the operator.

---

## Mesh Generation

The mesh is:

* Structured (i, j indexing)
* Non-uniform (geometric stretching)
* Divided into three regions:

  * Upstream (stretched)
  * Airfoil region (uniform)
  * Downstream (stretched)

Stretching factors:

* `XSF` → controls streamwise spacing
* `YSF` → controls normal clustering near the airfoil

---

## Boundary Conditions

* Far-field:
  [
  \phi = U_\infty x
  ]
* Symmetry (outside airfoil):
  [
  \frac{\partial \phi}{\partial y} = 0
  ]
* Airfoil surface:

  * Imposed normal velocity condition using airfoil thickness model

---

## How to Run

1. Configure simulation parameters in `main.rs`:

```rust
let config = Config {
    IMAX: 41,
    JMAX: 12,
    ILE: 10,
    ITE: 30,
    XSF: 1.25,
    YSF: 1.25,
    u_inf: 1.0,
    t: 0.05,
    n_max: 1000,
};
```

2. Choose the iterative scheme:

```rust
let tm_scheme = SOR { omega: 1.8 };
```

3. Run the solver:

```bash
cargo run --release
```

---

## Output Files

Generated in `job_files/`:

### Mesh

* `mesh.csv`

### Solution

* `solution.csv` (x, y, φ, u, v, Cp)

### Field Matrices

* `phi_matrix.csv`
* `cp_matrix.csv`
* `u_matrix.csv`
* `v_matrix.csv`

### Airfoil Data

* `airfoil_cp.csv`

### Convergence

* `residual_history.csv`

---

## Key Observations

* Jacobi converges slowly due to purely explicit updates
* Gauss-Seidel improves convergence via in-place updates
* SOR significantly accelerates convergence with optimal ω
* Mesh stretching strongly influences convergence behavior

---

## Future Work

* Implement Line Gauss-Seidel (tridiagonal solver per line)
* Add optimal relaxation parameter estimation
* Introduce multigrid acceleration
* Extend to compressible or viscous flows

---

## Requirements

* Rust (latest stable)
* Crates:

  * `ndarray`

---

## Author

CFD solver developed as part of an academic project in Aeronautical Engineering.

---

## License

This project is for educational purposes.
