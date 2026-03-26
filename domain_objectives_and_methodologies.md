# Domain Objectives & Methodologies
## Multi-Agent Deep Reinforcement Learning for Sustainable Adaptive Scheduling and Replanning in Dynamic Environments

---

## 🏆 TOP 1 DOMAIN: Project Management Scheduling

### Why This Is the Easiest Domain for a Student
- Minimal simulation complexity (no physics, no hardware)
- Small, interpretable state and action spaces
- No domain expertise required
- Free, well-documented benchmark datasets available (PSPLIB)
- Direct reward signal (minimize delay)
- Plug-and-play with Stable-Baselines3 / RLlib

---

### 📋 ALL OBJECTIVES — Project Management Scheduling

#### Objective 1: Minimize Total Project Makespan
- **Goal**: Complete all tasks in the shortest possible total time
- **Metric**: Makespan = time of last task completion
- **Why Easy**: Single scalar to minimize, straightforward reward shaping

#### Objective 2: Meet Task Deadlines (Minimize Tardiness)
- **Goal**: Complete each task by its individual deadline
- **Metric**: Total tardiness = sum of (completion_time − deadline) for late tasks
- **Why Easy**: Per-task penalty clearly maps to reward function

#### Objective 3: Minimize Resource Overload / Leveling
- **Goal**: Spread resource usage evenly over time (avoid peaks and idle periods)
- **Metric**: Resource utilization variance across time windows
- **Why Moderate**: Needs multi-objective reward balancing

#### Objective 4: Maximize Resource Utilization
- **Goal**: Keep workers/machines busy as much as possible
- **Metric**: Percentage of time resources are actively working
- **Complementary**: Often paired with Objective 3

#### Objective 5: Minimize Cost
- **Goal**: Reduce labor/resource cost while finishing on time
- **Metric**: Total cost = sum(resource cost × usage time)
- **Why Moderate**: Requires cost model per resource type

#### Objective 6: Adaptive Replanning Under Disruptions
- **Goal**: When a task is delayed or a resource fails, reschedule remaining tasks optimally
- **Metric**: Recovery time + post-disruption tardiness
- **Core to MARL Thesis**: This maps directly to your research topic

#### Objective 7: Sustainability / Energy Efficiency (Green Scheduling)
- **Goal**: Minimize energy consumed by scheduling tasks during off-peak hours or assigning them to low-power resources
- **Metric**: Total energy consumed
- **Relevant to Thesis**: "Sustainable" is in your research title

---

### 🔬 METHODOLOGIES — Project Management Scheduling

#### Methodology 1: Deep Q-Network (DQN)
| Component | Detail |
|---|---|
| **State** | Task status vector (pending/active/done), resource availability, time remaining |
| **Action** | Discrete: assign task T_i to resource R_j |
| **Reward** | −1 per time unit of delay; 0 when all tasks complete on time |
| **Network** | 2-layer MLP (128→64→|actions|) |
| **Replay Buffer** | Experience replay with size 10,000 |
| **Target Network** | Updated every 500 steps |
| **Best For** | Small projects (≤15 tasks, ≤3 resources) |

**Algorithm Steps:**
```
1. Observe state s_t (task statuses, resource loads, deadlines)
2. ε-greedy: pick action a_t (which task to assign, to which resource)
3. Execute action → get next state s_{t+1} and reward r_t
4. Store (s_t, a_t, r_t, s_{t+1}) in replay buffer
5. Sample minibatch → compute TD target → update Q-network
6. Repeat until convergence
```

---

#### Methodology 2: Proximal Policy Optimization (PPO)
| Component | Detail |
|---|---|
| **State** | Same as DQN + add resource utilization percentage |
| **Action** | Can handle both discrete (task selection) and continuous (priority weights) |
| **Reward** | Shaped: −tardiness − λ×makespan + bonus for on-time completion |
| **Actor Network** | MLP outputting action probabilities |
| **Critic Network** | MLP outputting state value V(s) |
| **Clip Ratio** | ε = 0.2 (standard PPO) |
| **Best For** | Medium projects, more stable training than DQN |

**Algorithm Steps:**
```
1. Collect N steps of experience using current policy π_θ
2. Compute advantage estimates Â_t using GAE (λ=0.95, γ=0.99)
3. Optimize clipped surrogate objective:
   L = E[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)]
4. Update value function: minimize (V_θ(s) − V_target)²
5. Repeat for K epochs per data collection phase
```

---

#### Methodology 3: Multi-Agent RL (MAPPO / MADDPG) — Core Thesis Method
| Component | Detail |
|---|---|
| **Agents** | Each resource (worker/machine) is an independent agent |
| **State** | Local: own task queue + global: all resource loads (shared critic) |
| **Action** | Each agent decides which pending task to pick up next |
| **Reward** | Global: shared team reward = −total_makespan; Local: individual resource utilization |
| **Coordination** | Centralized training, decentralized execution (CTDE) paradigm |
| **Best For** | Large projects with 5+ resources; directly supports MARL thesis |

**CTDE Architecture:**
```
Training Phase:
  ← All agents share observations to a central critic
  ← Central critic computes V(s^1, s^2, ..., s^n) for better gradient estimates

Execution Phase:
  → Each agent acts using only its LOCAL observation
  → No communication needed at runtime
```

---

#### Methodology 4: Graph Neural Network + RL (GNN-RL)
| Component | Detail |
|---|---|
| **State Representation** | Task dependency graph (nodes = tasks, edges = precedence constraints) |
| **Encoding** | GNN encodes graph structure → embedding fed into policy network |
| **Action** | Select node (task) to schedule next |
| **Best For** | Projects with complex task dependencies |
| **Why Advanced** | Requires PyTorch Geometric; higher complexity |

---

#### Methodology 5: Heuristic Baseline (for comparison)
Used as benchmark to evaluate RL agent performance:
- **Earliest Deadline First (EDF)**: Always schedule the task with the nearest deadline
- **Critical Path Method (CPM)**: Prioritize tasks on the longest dependency chain
- **Shortest Processing Time (SPT)**: Schedule shortest tasks first to maximize throughput

---

### 📦 Tech Stack for Top 1
```bash
pip install gymnasium stable-baselines3 numpy matplotlib pandas networkx
```

### 📁 Dataset
- **PSPLIB**: http://www.om-db.wi.tum.de/psplib/
- **J30 set**: 480 instances, 30 tasks each — perfect for students

---
---

## 🥈 TOP 2 DOMAIN: Cloud Computing Task Scheduling

### Why This Is the Second Easiest Domain for a Student
- Well-defined environment (servers = resources, jobs = tasks)
- Real-world datasets publicly available (Google Cluster Trace, Alibaba Cluster Trace)
- Many open-source gym environments exist
- Scales naturally from simple to complex
- Direct industry relevance

---

### 📋 ALL OBJECTIVES — Cloud Computing Task Scheduling

#### Objective 1: Minimize Job Completion Time (JCT)
- **Goal**: Complete each submitted job as fast as possible
- **Metric**: Average Job Completion Time across all jobs
- **Why Easy**: Single metric, clear reward signal

#### Objective 2: Maximize Cluster Utilization
- **Goal**: Keep CPU and memory resources occupied (reduce idle capacity)
- **Metric**: Average resource utilization = (used resources / total resources) × 100%
- **Common Tradeoff**: High utilization can increase JCT due to contention

#### Objective 3: Minimize Resource Fragmentation
- **Goal**: Avoid situations where leftover CPU/RAM is too small to fit any job (wasted holes)
- **Metric**: Fragmentation ratio = unusable leftover capacity / total capacity
- **Why Important**: Directly affects profit in commercial clouds

#### Objective 4: Energy-Aware Scheduling (Green Cloud)
- **Goal**: Minimize total energy by packing jobs onto fewer servers and powering others off
- **Metric**: Total kWh consumed per batch of jobs
- **Relevant to Thesis**: Matches "Sustainable" in research title

#### Objective 5: Fairness (Multi-Tenant Scheduling)
- **Goal**: Ensure no single user's jobs monopolize the cluster
- **Metric**: Jain's Fairness Index across tenants
- **Relevant for**: Multi-agent fairness between competing scheduling agents

#### Objective 6: Deadline-Constrained Scheduling (SLA)
- **Goal**: Complete jobs before their Service Level Agreement (SLA) deadlines
- **Metric**: SLA violation rate (percentage of jobs missing deadline)
- **Maps to**: Tardiness objective in Project Management

#### Objective 7: Adaptive Reschedules Under Node Failures
- **Goal**: When a server fails mid-execution, migrate/reschedule affected jobs with minimal delay
- **Metric**: Recovery time + additional JCT after failure
- **Core to MARL Thesis**: Dynamic replanning in changing environment

#### Objective 8: Cost Minimization (Spot Instance / Preemptible VMs)
- **Goal**: Use cheapest available resources (spot instances) while meeting deadlines
- **Metric**: Total dollar cost per job batch
- **Cloud-Specific**: Unique to cloud domain, not present in project management

---

### 🔬 METHODOLOGIES — Cloud Computing Task Scheduling

#### Methodology 1: Deep Q-Network (DQN) for Job Scheduling
| Component | Detail |
|---|---|
| **State** | Job queue (resource demands + deadlines) + server state (CPU%, RAM%) |
| **Action** | Assign job J_i to server S_j (discrete) |
| **Reward** | +10 for on-time completion; −1 per time step of waiting; −20 for SLA violation |
| **Network** | 3-layer MLP or CNN over resource matrix |
| **Exploration** | ε-greedy with ε decay from 1.0 → 0.01 over 50,000 steps |
| **Best For** | Single-cluster scheduling with ≤100 servers |

---

#### Methodology 2: Actor-Critic (A3C / A2C)
| Component | Detail |
|---|---|
| **Architecture** | Asynchronous multiple workers exploring in parallel |
| **State** | Resource utilization matrix (servers × resource_types) + job priority queue |
| **Action** | Which server to place next job on |
| **Advantage** | Lower variance than DQN; faster convergence |
| **Best For** | Medium clusters with continuous resource dimensions |

---

#### Methodology 3: Multi-Agent PPO (MAPPO) — Core Thesis Method
| Component | Detail |
|---|---|
| **Agents** | One RL agent per cluster zone / rack |
| **Local Obs** | Agent sees only its own rack's server states + pending jobs |
| **Global Critic** | Sees all rack states during training |
| **Action** | Each rack agent decides which local server to assign incoming jobs to |
| **Reward** | Shared: −total_cluster_JCT + Individual: −own_zone_utilization_variance |
| **CTDE** | Centralized training, decentralized execution |

**Multi-Agent Interaction:**
```
Zone Agent 1 ──┐
Zone Agent 2 ──┼──► Central Critic (training only) ──► Global Value V(s^all)
Zone Agent 3 ──┘

At execution: each zone acts independently using local obs only
```

---

#### Methodology 4: Attention-Based Pointer Network (Seq2Seq + RL)
| Component | Detail |
|---|---|
| **Input** | Sequence of jobs (each represented as feature vector) |
| **Architecture** | Transformer encoder + attention-based decoder |
| **Output** | Ordered sequence of (job → server) assignments |
| **Training** | REINFORCE algorithm with baseline |
| **Best For** | Batch job scheduling where order matters |
| **Why Advanced** | Requires transformer knowledge; higher implementation effort |

---

#### Methodology 5: Curriculum Learning for Dynamic Environments
| Stage | Description |
|---|---|
| **Stage 1** | Train on small, static cluster (10 servers, 20 jobs, no failures) |
| **Stage 2** | Add random job arrival rates (Poisson process) |
| **Stage 3** | Add server failures (10% failure probability per hour) |
| **Stage 4** | Full dynamic environment: failures + heterogeneous servers + SLA constraints |
| **Why Useful** | Prevents agent from being overwhelmed by full complexity immediately |
| **Maps to Thesis** | "Adaptive" and "Replanning" aspects of the research title |

---

#### Methodology 6: Heuristic Baselines (for comparison)
| Heuristic | Description |
|---|---|
| **First Fit** | Place job on first server with enough capacity |
| **Best Fit** | Place job on server with least remaining capacity after placement |
| **Round Robin** | Distribute jobs evenly across all servers in rotation |
| **Dominant Resource Fairness (DRF)** | Fair multi-resource allocation algorithm (Google-style) |
| **Tetris** | Pack jobs to minimize resource fragmentation (academic baseline) |

---

### 📦 Tech Stack for Top 2
```bash
pip install gymnasium stable-baselines3 numpy matplotlib pandas torch
# Optional: gym-cloudscheduling, OR-Gym
```

### 📁 Datasets
| Dataset | Source | Size |
|---|---|---|
| **Google Cluster Trace v3** | Google Research | 29-day trace, 12,500 machines |
| **Alibaba Cluster Trace 2018** | Alibaba Research | 4,000 machines, millions of tasks |
| **Synthetic Generator** | Custom Python script | Fully controllable |

---
---

## 📊 HEAD-TO-HEAD COMPARISON: Top 1 vs Top 2

| Feature | 🥇 Project Management | 🥈 Cloud Computing |
|---|---|---|
| **Environment Complexity** | ⭐ Very Simple | ⭐⭐ Simple |
| **State Space Size** | Small (10–100 dims) | Medium (100–10,000 dims) |
| **Action Space** | Small discrete | Medium discrete |
| **Data Availability** | PSPLIB (structured) | Google/Alibaba traces (huge) |
| **Domain Knowledge Needed** | None | Basic cloud concepts |
| **Setup Time** | 1–2 hours | 2–4 hours |
| **Existing Gym Envs** | Build from scratch (easy) | Several available |
| **MARL Applicability** | ✅ Each resource = agent | ✅ Each rack/zone = agent |
| **Sustainability Objective** | ✅ Energy-aware scheduling | ✅ Green cloud / server sleep |
| **Dynamic Replanning** | ✅ Task disruptions | ✅ Node failures |
| **Thesis Alignment** | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐⭐ Perfect |

---

## 🎯 Recommended Thesis Structure Using Both Domains

```
Chapter 1: Introduction
  → Problem: Dynamic scheduling in changing environments
  → Motivation: Energy efficiency + deadline compliance

Chapter 2: Literature Review
  → Traditional scheduling (CPM, EDF, Tetris)
  → DRL for scheduling (DQN, PPO, MARL)

Chapter 3: Methodology
  → MARL framework (CTDE, MAPPO)
  → Environment design for Project Management (Domain 1)
  → Environment design for Cloud Computing (Domain 2)

Chapter 4: Experiments — Domain 1 (Project Management)
  → Objectives 1–7 evaluated
  → DQN vs PPO vs MAPPO vs EDF baseline

Chapter 5: Experiments — Domain 2 (Cloud Computing)
  → Objectives 1–8 evaluated
  → MAPPO vs DRF vs Tetris baseline

Chapter 6: Cross-Domain Analysis
  → Which MARL architecture generalizes across both domains?
  → Sustainability gains in both domains

Chapter 7: Conclusion
```

---

## ⚡ Quick-Start Code Skeleton (Project Management — Top 1)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ProjectSchedulingEnv(gym.Env):
    """
    Simple Project Scheduling Environment for MARL research.
    Each resource is a separate agent (MARL mode) or single agent for baseline.
    """
    def __init__(self, n_tasks=10, n_resources=3, max_time=50):
        super().__init__()
        self.n_tasks = n_tasks
        self.n_resources = n_resources
        self.max_time = max_time

        # State: [task_status(n), resource_busy_until(n_resources), current_time(1)]
        self.observation_space = spaces.Box(
            low=0, high=max_time,
            shape=(n_tasks + n_resources + 1,),
            dtype=np.float32
        )
        # Action: assign task i to resource j → flattened index
        self.action_space = spaces.Discrete(n_tasks * n_resources)

    def reset(self, seed=None):
        self.task_status = np.zeros(self.n_tasks)       # 0=pending, 1=done
        self.task_duration = np.random.randint(1, 10, self.n_tasks)
        self.task_deadline = np.random.randint(10, self.max_time, self.n_tasks)
        self.resource_free_at = np.zeros(self.n_resources)
        self.current_time = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.task_status,
            self.resource_free_at,
            [self.current_time]
        ]).astype(np.float32)

    def step(self, action):
        task_idx = action // self.n_resources
        resource_idx = action % self.n_resources

        reward = 0
        if self.task_status[task_idx] == 0:  # task is pending
            start_time = max(self.current_time, self.resource_free_at[resource_idx])
            finish_time = start_time + self.task_duration[task_idx]
            self.resource_free_at[resource_idx] = finish_time
            self.task_status[task_idx] = 1  # mark done

            # Reward: penalize tardiness
            tardiness = max(0, finish_time - self.task_deadline[task_idx])
            reward = -tardiness
        else:
            reward = -5  # penalty for invalid action

        self.current_time += 1
        done = bool(np.all(self.task_status == 1) or self.current_time >= self.max_time)
        return self._get_obs(), reward, done, False, {}


# Training with PPO (Stable-Baselines3)
from stable_baselines3 import PPO

env = ProjectSchedulingEnv(n_tasks=10, n_resources=3)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048)
model.learn(total_timesteps=100_000)
model.save("project_scheduler_ppo")
print("Training complete!")
```

---

*Generated for: MULTI-AGENT DEEP REINFORCEMENT LEARNING FOR SUSTAINABLE ADAPTIVE SCHEDULING AND REPLANNING IN DYNAMIC ENVIRONMENTS*
