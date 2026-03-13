# Multi-Objective Genetic Algorithm (MOGA) — Detailed Explanation with Flowchart

---

## 1. What is a Multi-Objective Genetic Algorithm?

A **Multi-Objective Genetic Algorithm (MOGA)** is an evolutionary optimization technique inspired by biological evolution (natural selection, crossover, mutation). Unlike single-objective optimization (which minimizes/maximizes one value), MOGA simultaneously optimizes **two or more conflicting objectives**.

### Classic Examples
| Problem | Objective 1 | Objective 2 |
|---|---|---|
| Engineering design | Minimize weight | Maximize strength |
| Finance | Maximize return | Minimize risk |
| Scheduling | Minimize cost | Minimize time |

---

## 2. Key Concepts

### Pareto Dominance
A solution **A dominates** solution **B** if:
- A is **at least as good** as B on **all** objectives, AND
- A is **strictly better** than B on **at least one** objective.

### Pareto Front / Pareto-Optimal Set
The set of all non-dominated solutions is called the **Pareto front**. No solution in the Pareto front is better than another across all objectives — each represents a different trade-off.

### Fitness Assignment in MOGA
Instead of a single fitness value, fitness is determined by:
1. **Rank** — based on how many solutions dominate a given solution (lower rank = better).
2. **Crowding Distance** — measures how spread out solutions are in the objective space (higher = better diversity).

---

## 3. Popular MOGA Variants
- **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) — most widely used
- **SPEA2** (Strength Pareto Evolutionary Algorithm 2)
- **MOEA/D** (Multi-Objective Evolutionary Algorithm based on Decomposition)
- **PAES** (Pareto Archived Evolution Strategy)

---

## 4. General Flowchart of a Multi-Objective GA

```
┌─────────────────────────────────────────────────────────────────┐
│                         START                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Define the Problem                                     │
│  • Identify decision variables (x₁, x₂, ..., xₙ) — n vars      │
│  • Define objective functions f₁(x), f₂(x), ..., fₘ(x) — m obj│
│  • Set constraints (if any)                                     │
│  • Define variable bounds / search space                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Set Algorithm Parameters                               │
│  • Population size (N)                                          │
│  • Maximum generations (G_max)                                  │
│  • Crossover probability (p_c)                                  │
│  • Mutation probability (p_m)                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Initialize Population                                  │
│  • Randomly generate N candidate solutions                      │
│  • Each solution is a chromosome encoding decision variables    │
│  • Evaluate all objectives for each solution                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Evaluate Fitness                                       │
│  • Compute f₁(x), f₂(x), ..., fₘ(x) for all individuals       │
│  • Perform Non-dominated Sorting → assign rank to each          │
│    individual based on Pareto dominance                         │
│    (Rank 1 = non-dominated, Rank 2 = dominated only by         │
│     Rank-1 solutions, etc.)                                     │
│  • Calculate Crowding Distance for diversity preservation       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Selection                                              │
│  • Use tournament selection or rank-based selection             │
│  • Prefer solutions with:                                       │
│    (a) Lower Pareto rank (better dominance)                     │
│    (b) Higher crowding distance (better diversity) if equal rank│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Crossover (Recombination)                              │
│  • With probability p_c, apply crossover operator               │
│  • Common methods:                                              │
│    – Single-point / Two-point crossover (binary encoding)       │
│    – Simulated Binary Crossover — SBX (real-valued encoding)    │
│  • Produces two offspring from two parent solutions             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Mutation                                               │
│  • With probability p_m, apply mutation operator                │
│  • Common methods:                                              │
│    – Bit-flip mutation (binary encoding)                        │
│    – Polynomial Mutation (real-valued encoding)                 │
│  • Introduces genetic diversity to avoid premature convergence  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Evaluate Offspring                                     │
│  • Compute all objective function values for each new offspring │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: Merge & Environmental Selection                        │
│  • Combine parent pop (Pₜ, gen t) and offspring pop (Qₜ, gen t)│
│    → Combined pool R = Pₜ ∪ Qₜ  (size 2N)                      │
│  • Apply Non-dominated Sorting on R                             │
│  • Fill next generation Pₜ₊₁ (gen t+1, size N) by:             │
│    1. Adding front-by-front (best ranks first)                  │
│    2. If a front partially fits, sort by crowding distance      │
│       and select the most spread-out solutions                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────┴──────────────┐
              │  Termination Condition Met?  │
              │  (generation ≥ G_max, or     │
              │   convergence criterion)     │
              └──────────────┬──────────────┘
                   NO │               │ YES
                      │               │
                      ▼               ▼
             ┌────────────┐  ┌────────────────────────────────────┐
             │  t = t + 1 │  │  STEP 10: Extract Pareto Front     │
             │  Go back   │  │  • The final non-dominated set is  │
             │  to Step 5 │  │    the approximate Pareto front    │
             └────────────┘  │  • Each solution = a trade-off     │
                             │    between all objectives          │
                             └─────────────────┬──────────────────┘
                                               │
                                               ▼
                             ┌─────────────────────────────────────┐
                             │  STEP 11: Decision Making           │
                             │  • Present Pareto front to the      │
                             │    decision maker                   │
                             │  • Select the preferred solution    │
                             │    based on domain expertise /      │
                             │    preference criteria              │
                             └─────────────────┬───────────────────┘
                                               │
                                               ▼
                             ┌─────────────────────────────────────┐
                             │                END                  │
                             └─────────────────────────────────────┘
```

---

## 5. Step-by-Step Explanation

### Step 1 — Problem Definition
Define what you want to optimize. For example:
- Minimize cost `f₁(x)` and minimize delivery time `f₂(x)` simultaneously.
- Specify the variables (routes, quantities, etc.) and any constraints.

### Step 2 — Parameter Setting
Choose algorithm parameters:
- **Population size** typically ranges from 50–200.
- **Crossover probability** ~ 0.8–0.9.
- **Mutation probability** ~ 0.01–0.1.
- **Number of generations** depends on problem complexity.

### Step 3 — Initialization
- Generate N random solutions within the search space.
- Each solution is a *chromosome* (array of genes representing the variables).

### Step 4 — Non-dominated Sorting & Crowding Distance
This is the **core of MOGA** (especially NSGA-II):
1. **Non-dominated Sorting**: Assign each solution a rank:
   - Rank 1: solutions not dominated by any other solution.
   - Rank 2: solutions dominated only by Rank-1 solutions.
   - Continue until all solutions are ranked.
2. **Crowding Distance**: For each solution in the same rank, calculate the average distance to its neighbors in objective space. Higher crowding distance = more isolated = more diverse.

### Step 5 — Selection
Use **binary tournament selection**:
- Randomly pick 2 solutions.
- The winner has lower rank, or if same rank, higher crowding distance.
- Repeat until enough parents are selected.

### Step 6 — Crossover
- Randomly pair parents.
- Exchange genetic information to create offspring.
- **SBX (Simulated Binary Crossover)** is commonly used for real-valued variables.

### Step 7 — Mutation
- Randomly alter genes of offspring with small probability `p_m`.
- **Polynomial Mutation** is standard for real-valued representations.
- Ensures the algorithm does not get stuck in local optima.

### Step 8 — Offspring Evaluation
- Evaluate all objective functions for the new offspring.

### Step 9 — Merging and Next-Generation Selection (Elitism)
- Combine parent and offspring populations into a pool of size 2N.
- Re-sort using non-dominated sorting.
- Select the best N solutions for the next generation.
- This **elitist** strategy ensures good solutions are not lost.

### Step 10 — Final Pareto Front
- When the algorithm terminates, the Rank-1 solutions in the final population form the **approximate Pareto-optimal front**.
- This is a set of trade-off solutions covering the entire objective space.

### Step 11 — Decision Making
- The decision maker selects the final solution from the Pareto front based on preferences or utility.

---

## 6. Objective Space Visualization

```
  f₂ (Objective 2 — e.g., Cost)
  ↑
  │  ×         ×
  │    ×  ×
  │       ○ ← Pareto Front (non-dominated solutions)
  │     ○    ○
  │   ○         ○
  │ ○              ○
  └──────────────────────→ f₁ (Objective 1 — e.g., Time)

  ○ = Pareto-optimal solutions (trade-off curve)
  × = Dominated solutions (can be improved in at least one objective)
```

---

## 7. Advantages and Limitations

### Advantages
- Can find a **set of Pareto-optimal solutions** in a single run.
- Works well for **non-convex, discontinuous, or multimodal** Pareto fronts.
- No need to aggregate objectives into a single function.
- Naturally maintains **diversity** via crowding distance.

### Limitations
- Computationally expensive for **large populations** or **many objectives** (many-objective optimization).
- Parameter tuning (population size, crossover/mutation rates) can be problem-specific.
- Quality of results depends on **problem encoding** (chromosome representation).
- May struggle with **more than 3–4 objectives** (the "curse of dimensionality" — use MOEA/D or NSGA-III for many objectives).

---

## 8. Summary Table

| Phase | What Happens |
|---|---|
| Initialization | Random population generated |
| Evaluation | Objective values computed for all solutions |
| Non-dominated Sorting | Assign Pareto rank to each solution |
| Crowding Distance | Measure spread of solutions in objective space |
| Selection | Tournament selection using rank + crowding |
| Crossover | Parent genes recombined to produce offspring |
| Mutation | Small random changes in offspring genes |
| Merging | Parents + offspring combined into 2N pool |
| Environmental Selection | Best N selected for next generation (elitism) |
| Termination | Repeat until max generations or convergence |
| Output | Approximate Pareto front of trade-off solutions |

---

*This document explains the general working of a Multi-Objective Genetic Algorithm (MOGA), particularly the widely-used NSGA-II framework, with a detailed step-by-step flowchart.*
