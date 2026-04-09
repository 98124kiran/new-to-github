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

---
---

## 🥉 TOP 3 DOMAIN: Edge Computing Task Offloading & Scheduling

### ✅ Is Edge Computing Easy for a Student? — Verdict: YES (Recommended)

**Edge Computing ranks as the 3rd easiest domain** for this exact MARL research thesis. Here is why:

| Factor | Assessment |
|---|---|
| **Action space** | Very small: {process locally, offload to edge server 1, offload to edge server 2, offload to cloud} |
| **State space** | Manageable: task size + device battery + server load + network bandwidth |
| **Domain knowledge** | Minimal — no hardware needed, fully simulatable |
| **Open-source envs** | Several available (MEC-Gym, EdgeSimPy, custom gym wrappers) |
| **Real-world relevance** | Extremely high: IoT, smart cities, healthcare wearables, autonomous vehicles |
| **MARL fit** | ⭐⭐⭐⭐⭐ Perfect — each IoT device = one independent agent |
| **Thesis keyword fit** | ⭐⭐⭐⭐⭐ Every word in your title maps naturally to edge computing |

**Why It Fits Your Thesis Title Perfectly:**
```
"Multi-Agent"   → Each IoT device / mobile device = 1 independent RL agent
"Deep RL"       → DQN / PPO policy for offloading decisions
"Sustainable"   → Minimize device battery drain + edge server energy consumption
"Adaptive"      → React to changing network conditions and server loads in real time
"Replanning"    → When an edge server goes offline, reroute tasks to another node
"Dynamic"       → Network bandwidth fluctuates, servers get overloaded, tasks arrive unpredictably
"Environment"   → The network topology with IoT devices, edge nodes, and cloud backend
```

---

### 🌍 Real-World Scenarios for Edge Computing

#### Scenario 1: Smart Healthcare — Hospital Wearables
```
Real World:
  Patient wears a health monitor (ECG, SpO2, blood pressure sensor)
  Monitor generates tasks: "analyze ECG pattern", "detect anomaly"

Problem:
  Device has limited battery → cannot process everything locally
  Must decide: process on device OR offload to hospital edge server OR send to cloud

MARL Application:
  Each patient device = 1 agent
  Agents compete for hospital edge server bandwidth
  Goal: minimize patient data analysis latency + preserve device battery life
  Dynamic disruption: edge server goes offline → agents replan to use cloud
```

#### Scenario 2: Autonomous Vehicles — Roadside Edge Nodes
```
Real World:
  Self-driving cars generate massive real-time tasks: object detection, lane recognition,
  pedestrian tracking, traffic sign reading

Problem:
  Car's onboard GPU is limited → cannot run all models at 30fps locally
  Must offload to roadside edge nodes (RSUs) for low-latency response

MARL Application:
  Each vehicle = 1 agent making offloading decisions at every intersection
  Agents coordinate to avoid overloading the same RSU simultaneously
  Dynamic disruption: RSU failure → car must switch to next RSU or process locally
  Sustainability: minimize total computation energy across the vehicle fleet
```

#### Scenario 3: Smart Manufacturing — Factory Floor
```
Real World:
  Robotic arms on a factory line run quality inspection using computer vision
  Defect detection tasks need to be processed within milliseconds

Problem:
  Factory has 50 robots generating visual inspection tasks simultaneously
  Edge servers on each production line have limited capacity

MARL Application:
  Each robot = 1 agent deciding which edge server to send its inspection task to
  Goal: minimize inspection latency + balance edge server loads
  Sustainability: schedule heavy tasks during energy off-peak hours
  Dynamic disruption: production surge → edge servers overloaded → adaptive rescheduling
```

#### Scenario 4: Smart City — Traffic Camera Analytics
```
Real World:
  Hundreds of traffic cameras send video to nearby edge nodes for:
  - License plate recognition
  - Congestion detection
  - Accident detection

Problem:
  At peak hours, all cameras generate tasks simultaneously
  Edge nodes become overloaded → need to offload some tasks to cloud

MARL Application:
  Each camera = 1 agent deciding when to process locally vs edge vs cloud
  Goal: minimize detection latency (accidents must be detected in <2 seconds)
  Sustainability: minimize total network data transfer + edge server energy
```

#### Scenario 5: Disaster Response — First Responder Networks
```
Real World:
  Firefighters/rescue teams carry body cameras and sensors in disaster zones
  Edge servers mounted on response vehicles process situational awareness tasks

Problem:
  Vehicle edge servers have limited power (running on generator)
  Network coverage is patchy and unreliable

MARL Application:
  Each responder device = 1 agent
  Adaptive replanning: when one vehicle leaves the area, reroute tasks to another
  Sustainability: maximize task processing per unit of generator fuel
```

---

### 📋 ALL OBJECTIVES — Edge Computing Task Offloading

#### Objective 1: Minimize Task Execution Latency (E2E Delay)
- **Goal**: Complete each task (local processing + transmission + edge processing) in minimum time
- **Metric**: End-to-end latency = transmission delay + queuing delay + processing time
- **Real World**: Autonomous vehicle must detect pedestrians in <100ms to stop safely
- **Why Easy**: Single scalar to minimize; direct reward signal

#### Objective 2: Minimize Device Energy Consumption (Battery Life)
- **Goal**: Reduce energy used by IoT/mobile device for local processing and data transmission
- **Metric**: Energy = CPU power × local processing time + transmission power × data size / bandwidth
- **Real World**: Hospital wearable should last 24 hours on one charge
- **Sustainability Link**: Directly matches "Sustainable" in thesis title

#### Objective 3: Minimize Edge Server Energy Consumption (Green Edge)
- **Goal**: Reduce total energy consumed by edge servers — turn off idle servers, consolidate tasks
- **Metric**: Total server energy = sum(active server power × active time)
- **Real World**: A network of 500 edge nodes in a smart city consumes as much power as a small data center
- **Sustainability Link**: This is the "green" angle unique to edge computing

#### Objective 4: Meet Task Deadline Constraints
- **Goal**: Ensure time-critical tasks (medical alerts, collision detection) finish before their deadlines
- **Metric**: Deadline miss rate (%) + weighted tardiness for near-miss tasks
- **Real World**: ECG anomaly alert must trigger within 5 seconds to be clinically useful
- **Maps to**: SLA objective in Cloud Computing; Tardiness in Project Management

#### Objective 5: Maximize Edge Server Utilization
- **Goal**: Keep edge server compute capacity occupied without overloading
- **Metric**: Utilization = (tasks processed) / (total server capacity) per time window
- **Tradeoff**: High utilization increases queueing delay; needs balance

#### Objective 6: Minimize Data Privacy / Offloading Cost
- **Goal**: Sensitive tasks (medical, personal) should stay on local/edge devices rather than going to cloud
- **Metric**: Privacy violation score + monetary offloading cost to cloud provider
- **Real World**: GDPR compliance — patient ECG data must not leave hospital edge network
- **Unique to Edge**: Not present in cloud or project management domains

#### Objective 7: Adaptive Task Rerouting Under Edge Server Failure
- **Goal**: When an edge node goes offline, immediately reroute its queued tasks to neighboring nodes or cloud
- **Metric**: Service recovery time + additional latency after failure event
- **Real World**: Road-side unit (RSU) loses power → all vehicles suddenly need to offload elsewhere
- **Core to Thesis**: "Replanning in Dynamic Environment" — this is the central scenario

#### Objective 8: Load Balancing Across Edge Nodes (Multi-Agent Coordination)
- **Goal**: Prevent any one edge server from becoming a bottleneck while others are idle
- **Metric**: Standard deviation of server utilization across all edge nodes (minimize it)
- **Real World**: In a smart factory, one production line's edge server gets overwhelmed at shift start
- **MARL Specific**: Requires agents to coordinate without direct communication

---

### 🔬 METHODOLOGIES — Edge Computing Task Offloading

#### Methodology 1: DQN for Single-Device Offloading (Simplest Baseline)
| Component | Detail |
|---|---|
| **State** | [task_size, task_deadline, device_battery%, edge_server_load%, network_bandwidth] |
| **Action** | Discrete: {0=process_locally, 1=offload_edge1, 2=offload_edge2, 3=offload_cloud} |
| **Reward** | −latency − α×energy_cost − β×deadline_violation_penalty |
| **Network** | 2-layer MLP (64→32→4 actions) |
| **Training Steps** | 50,000–200,000 steps sufficient |
| **Best For** | Single device learning, baseline experiments |

**Algorithm Steps:**
```
1. Observe state s_t: [task_size=500KB, battery=78%, edge_load=45%, bandwidth=10Mbps]
2. ε-greedy: pick action a_t = "offload to edge server 1"
3. Simulate offloading: latency=120ms, energy_used=0.02J
4. Compute reward: r_t = −0.12 (latency) − 0.002 (energy) = −0.122
5. Store in replay buffer, update Q-network
6. Over time: agent learns to offload to edge when battery is low, process locally when edge is busy
```

---

#### Methodology 2: PPO for Multi-Task Scheduling on Edge
| Component | Detail |
|---|---|
| **State** | Queue of N pending tasks + edge server states + device battery |
| **Action** | For each task in queue: continuous priority weight OR discrete offload decision |
| **Reward** | −Σ(weighted latency) − γ×energy + δ×tasks_completed_on_time |
| **Architecture** | Actor-Critic MLP; actor outputs softmax over offload options |
| **Advantage** | Handles multiple tasks being scheduled simultaneously per timestep |
| **Best For** | Device with a queue of 5–20 tasks arriving in a burst (e.g., video frame burst) |

---

#### Methodology 3: MAPPO — Each IoT Device Is an Agent (Core Thesis Method)
| Component | Detail |
|---|---|
| **Agents** | Each IoT device / mobile user = 1 independent RL agent |
| **Local Observation** | Own task queue + own battery + visible edge server loads |
| **Global Critic** | During training: sees ALL devices' states + ALL server loads |
| **Action** | Each device decides where to offload its next task |
| **Shared Reward** | −total_system_latency − total_energy (global sustainability metric) |
| **Individual Reward** | −own_deadline_violations − own_battery_drain |
| **CTDE** | Centralized training, decentralized execution |

**Real-World CTDE Explanation:**
```
Training (done once, offline):
  Simulate 100 hospital wearables simultaneously
  Central critic sees ALL 100 devices' battery levels + hospital edge server load
  Each device agent learns: "if edge server load > 80%, don't offload there"

Deployment (runs forever, no central coordinator needed):
  Each wearable runs its own policy independently
  No Wi-Fi connection to a central server needed during operation
  Works even if the hospital network is partially down
```

**Multi-Agent Load Balancing Without Explicit Communication:**
```
Agent 1 (Device A) sees: edge_server_1_load = 60% → offloads there
Agent 2 (Device B) sees: edge_server_1_load = 75% → avoids, offloads to edge_server_2
Agent 3 (Device C) sees: edge_server_1_load = 90% → processes locally

Result: Implicit load balancing without agents talking to each other
This is the core MARL contribution of your thesis!
```

---

#### Methodology 4: Lyapunov Optimization + DRL (Hybrid — Advanced)
| Component | Detail |
|---|---|
| **Lyapunov Component** | Provides theoretical stability guarantees for queue management |
| **DRL Component** | Handles real-time offloading decisions under dynamic conditions |
| **Why Combine** | Lyapunov alone is rigid; DRL alone has no stability guarantees |
| **Best For** | Research papers requiring theoretical analysis + empirical results |
| **Complexity** | ⭐⭐⭐⭐ High — requires optimization theory background |
| **Use In Thesis** | Mention as "future work" or advanced variant |

---

#### Methodology 5: Curriculum Learning for Edge Environments
| Stage | Scenario Simulated | What Agent Learns |
|---|---|---|
| **Stage 1** | 1 device, 1 edge server, stable network, no failures | Basic offload/local decision |
| **Stage 2** | 5 devices, 2 edge servers, moderate contention | Basic load balancing |
| **Stage 3** | 20 devices, 3 edge servers, random bandwidth fluctuation | Adapt to dynamic network |
| **Stage 4** | 50 devices, 5 edge servers, random server failures, task bursts | Full MARL replanning |
| **Real World** | Mirrors gradual deployment: pilot → ward → hospital-wide | Validates scalability |

---

#### Methodology 6: Heuristic Baselines (Comparison Only)
| Heuristic | Decision Rule | Real-World Equivalent |
|---|---|---|
| **Always Local** | Never offload; process everything on device | Conservative, battery-killing approach |
| **Always Offload** | Always send to edge/cloud | Latency-heavy when server is congested |
| **Greedy Offload** | Offload if task size > threshold | Simple rule used in early IoT systems |
| **Least Loaded First** | Always offload to the server with lowest current load | Standard load balancer |
| **Random** | Randomly pick offload destination | Lower bound baseline |

---

### 🌐 Complete System Architecture — Edge Computing MARL

```
                    ┌─────────────────────────────────────────┐
                    │            INTERNET / CLOUD              │
                    │   (backup compute, high latency, costly) │
                    └──────────────────┬──────────────────────┘
                                       │ (high latency link)
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────┴──────┐             ┌──────┴──────┐             ┌──────┴──────┐
    │  Edge Node 1│             │  Edge Node 2│             │  Edge Node 3│
    │ (hospital   │             │ (factory    │             │ (roadside   │
    │  server)    │             │  floor)     │             │  RSU unit)  │
    └──────┬──────┘             └──────┬──────┘             └──────┬──────┘
           │ (low latency link)         │                           │
    ┌──────┼──────┐             ┌──────┼──────┐             ┌──────┼──────┐
    │      │      │             │      │      │             │      │      │
  📱Wearable   📱Wearable    🤖Robot  🤖Robot   🚗Car    🚗Car    🚗Car
  Agent 1    Agent 2       Agent 3  Agent 4   Agent 5  Agent 6  Agent 7

  Each 📱🤖🚗 = 1 MARL Agent making independent offloading decisions
  Each makes decisions using ONLY local observations (CTDE execution phase)
```

---

### 📦 Tech Stack for Top 3 (Edge Computing)
```bash
pip install gymnasium stable-baselines3 numpy matplotlib pandas torch
# Edge-specific simulation:
pip install EdgeSimPy          # edge computing simulator
# OR build lightweight custom gym env (recommended for students — 100-150 lines)
```

### 📁 Datasets & Benchmarks
| Dataset | Description | Use |
|---|---|---|
| **Shanghai Telecom Dataset** | Real mobile user traces + base station loads | Network load patterns |
| **Alibaba Mobile Edge Trace** | Task offloading traces from real deployments | Task size + arrival rates |
| **MEC Synthetic Generator** | Configurable Poisson task arrivals + random failures | Student experiments |
| **SUMO Traffic Simulator** | Vehicle mobility traces for V2X edge scenarios | Autonomous vehicle scenario |

---
---

## 📊 THREE-DOMAIN COMPARISON: Top 1 vs Top 2 vs Top 3

| Feature | 🥇 Project Management | 🥈 Cloud Computing | 🥉 Edge Computing |
|---|---|---|---|
| **Environment Complexity** | ⭐ Very Simple | ⭐⭐ Simple | ⭐⭐ Simple |
| **State Space Size** | Small (10–100 dims) | Medium (100–10,000 dims) | Small–Medium (10–500 dims) |
| **Action Space** | Small discrete | Medium discrete | Very small discrete (3–4 choices) |
| **Data Availability** | PSPLIB (structured) | Google/Alibaba traces | Shanghai Telecom / Synthetic |
| **Domain Knowledge Needed** | None | Basic cloud concepts | Basic IoT/network concepts |
| **Setup Time** | 1–2 hours | 2–4 hours | 2–3 hours |
| **Existing Gym Envs** | Build from scratch (easy) | Several available | A few (EdgeSimPy) + custom |
| **MARL Applicability** | ✅ Each resource = agent | ✅ Each rack/zone = agent | ✅ Each IoT device = agent |
| **Sustainability Objective** | ✅ Energy-aware scheduling | ✅ Green cloud / server sleep | ✅✅ Battery + server energy (dual) |
| **Dynamic Replanning** | ✅ Task disruptions | ✅ Node failures | ✅ Server failures + network changes |
| **Real-World Impact** | ⭐⭐⭐ Business projects | ⭐⭐⭐⭐ Industry infrastructure | ⭐⭐⭐⭐⭐ Healthcare/vehicles/cities |
| **Thesis Alignment** | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐⭐⭐ Perfect |
| **Student Recommendation** | ✅ Start here | ✅ Second experiment | ✅ Third experiment or swap with Top 2 |

---

## 🎯 Recommended Thesis Structure Using All Three Domains

```
Chapter 1: Introduction
  → Problem: Dynamic scheduling in changing environments
  → Motivation: Energy efficiency + deadline compliance + real-world IoT growth

Chapter 2: Literature Review
  → Traditional scheduling (CPM, EDF, Greedy Offload)
  → DRL for scheduling (DQN, PPO, MARL)
  → Edge computing + MARL papers (survey)

Chapter 3: Methodology
  → MARL framework (CTDE, MAPPO)
  → Environment design for Project Management (Domain 1)
  → Environment design for Cloud Computing (Domain 2)
  → Environment design for Edge Computing (Domain 3)

Chapter 4: Experiments — Domain 1 (Project Management)
  → Objectives 1–7 evaluated
  → DQN vs PPO vs MAPPO vs EDF baseline

Chapter 5: Experiments — Domain 2 (Cloud Computing)
  → Objectives 1–8 evaluated
  → MAPPO vs DRF vs Tetris baseline

Chapter 6: Experiments — Domain 3 (Edge Computing)
  → Objectives 1–8 evaluated across 3 real-world scenarios
  → MAPPO vs Greedy Offload vs Least Loaded baseline
  → Case study: Smart Healthcare scenario (most impactful)

Chapter 7: Cross-Domain Analysis
  → Which MARL architecture generalizes across all three domains?
  → Sustainability gains compared across domains
  → Scalability: project (10 tasks) → cloud (1000 jobs) → edge (50 devices)

Chapter 8: Conclusion
  → Contributions to MARL + sustainable scheduling
  → Limitations and future work
```

---

## ⚡ Quick-Start Code Skeletons

### Skeleton 1: Project Management (Top 1)

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

### Skeleton 2: Edge Computing Offloading (Top 3)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class EdgeOffloadingEnv(gym.Env):
    """
    Edge Computing Task Offloading Environment for MARL research.
    Simulates an IoT device deciding where to process each task:
      0 = process locally on device
      1 = offload to edge server 1
      2 = offload to edge server 2
      3 = offload to cloud

    Real-world analogy: hospital wearable deciding where to run ECG analysis.
    """
    def __init__(self, n_edge_servers=2, max_tasks_per_episode=20):
        super().__init__()
        self.n_edge_servers = n_edge_servers
        self.max_tasks = max_tasks_per_episode
        # Actions: 0=local, 1..n_edge_servers=edge, last=cloud
        self.n_actions = 1 + n_edge_servers + 1
        self.action_space = spaces.Discrete(self.n_actions)
        # State: [task_size_norm, task_deadline_norm, battery_level,
        #         edge1_load, edge2_load, bandwidth_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3 + n_edge_servers + 1,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        self.tasks_done = 0
        self.battery = 1.0          # 100% battery
        self.edge_loads = np.random.uniform(0.1, 0.5, self.n_edge_servers)
        return self._generate_task(), {}

    def _generate_task(self):
        """Generate a random task and return the state observation."""
        self.task_size = np.random.uniform(0.1, 1.0)       # normalized 0–1
        self.task_deadline = np.random.uniform(0.3, 1.0)   # normalized urgency
        self.bandwidth = np.random.uniform(0.2, 1.0)       # current network quality
        return np.array([
            self.task_size,
            self.task_deadline,
            self.battery,
            *self.edge_loads,
            self.bandwidth
        ], dtype=np.float32)

    def step(self, action):
        latency, energy = self._simulate_offloading(action)

        # Reward: minimize latency and energy, penalize deadline violations
        deadline_threshold = self.task_deadline * 0.5  # urgency window
        deadline_penalty = -5.0 if latency > deadline_threshold else 0.0
        reward = -latency - 0.5 * energy + deadline_penalty

        # Update state
        self.battery = max(0.0, self.battery - energy * 0.05)
        # Edge server loads fluctuate dynamically
        self.edge_loads = np.clip(
            self.edge_loads + np.random.uniform(-0.1, 0.1, self.n_edge_servers),
            0.0, 1.0
        )
        self.tasks_done += 1
        done = self.tasks_done >= self.max_tasks or self.battery <= 0.0
        next_obs = self._generate_task() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return next_obs, reward, done, False, {
            "latency": latency, "energy": energy, "battery": self.battery
        }

    def _simulate_offloading(self, action):
        """
        Simulate latency and energy for each offloading decision.
        Returns (latency, energy_cost) both normalized 0–1.
        """
        if action == 0:
            # Process locally: fast if small task, slow if large; always uses battery
            latency = self.task_size * 0.8          # local CPU is slow
            energy = self.task_size * 0.6           # high local CPU energy
        elif action <= self.n_edge_servers:
            # Offload to edge server (action 1 or 2)
            server_idx = action - 1
            transmission = self.task_size / max(self.bandwidth, 0.1)
            queue_delay = self.edge_loads[server_idx] * 0.5
            latency = transmission + queue_delay
            energy = transmission * 0.3             # transmission energy only
        else:
            # Offload to cloud: high latency, low device energy, high cost
            latency = self.task_size / max(self.bandwidth, 0.1) + 0.4  # WAN delay
            energy = self.task_size * 0.2           # small transmission energy
        return float(np.clip(latency, 0, 1)), float(np.clip(energy, 0, 1))


# Training with PPO (Stable-Baselines3)
from stable_baselines3 import PPO

env = EdgeOffloadingEnv(n_edge_servers=2, max_tasks_per_episode=20)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024)
model.learn(total_timesteps=100_000)
model.save("edge_offloading_ppo")
print("Edge Computing agent training complete!")

# Quick evaluation
obs, _ = env.reset()
for _ in range(20):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    action_names = ["Local", "Edge-1", "Edge-2", "Cloud"]
    print(f"Action: {action_names[action]:8s} | Latency: {info['latency']:.3f} | "
          f"Energy: {info['energy']:.3f} | Battery: {info['battery']:.2f}")
    if done:
        break
```

---
---

## 🚗 SPECIALIZATION: Autonomous Vehicle (V2X) Edge Computing

### ❓ Is AV Edge Computing Easy for a Student? — Honest Assessment

**Short answer: YES — but only if you keep it simulation-based (no real cars, no hardware).**

Here is an honest breakdown of the difficulty:

| Factor | Generic Edge (Top 3) | AV-Specific Edge | Verdict |
|---|---|---|---|
| **Action space** | 3–4 choices | 3–5 choices (same!) | ✅ Same difficulty |
| **State space** | 5–6 variables | 7–9 variables (adds vehicle speed, position) | ✅ Still manageable |
| **Domain knowledge** | Basic IoT | Basic V2X/networking concepts | ⚠️ Small extra learning |
| **Simulation tools** | Custom gym (easy) | SUMO traffic sim OR custom gym | ✅ SUMO is free and well-documented |
| **Data availability** | Synthetic | SUMO synthetic / VeReMi dataset | ✅ Free & available |
| **Hardware needed** | None | None (pure simulation) | ✅ No hardware needed |
| **Real-world relevance** | High | Extremely high (Tesla, Waymo, industry) | ✅⭐ Best for CV/portfolio |
| **MARL fit** | Each IoT device = agent | Each vehicle = agent | ✅ Same MARL structure |
| **Thesis keyword fit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Perfect fit |

**Conclusion: Autonomous Vehicle edge computing is equally easy to implement as the generic edge scenario — because the MARL structure is identical. A vehicle is just a specific type of IoT agent.**

**The only extra step:** learn what tasks a vehicle generates (object detection, path planning, HD map update) — this takes 2–3 hours of reading, not weeks.

---

### 🌍 Real-World Context: Why Autonomous Vehicles Need Edge Computing

```
The Problem with Onboard-Only Processing:
  A self-driving car generates 4 TB of sensor data per day
  (cameras, LiDAR, radar, GPS, ultrasonic sensors)

  The car's onboard computer (e.g., NVIDIA DRIVE Orin) can handle some tasks,
  but NOT all tasks at low latency simultaneously:

  Task 1: Pedestrian detection (LiDAR + camera fusion)  → 50ms deadline
  Task 2: Lane change path planning                      → 100ms deadline
  Task 3: HD map update (download new road changes)      → 2000ms deadline
  Task 4: V2V collision warning (other car sends alert)  → 20ms deadline ← CRITICAL
  Task 5: Traffic signal phase prediction                → 500ms deadline

  Solution: OFFLOAD non-critical tasks to nearby Roadside Unit (RSU) edge servers
  Keep only safety-critical tasks (V2V collision) processed locally
```

```
The V2X (Vehicle-to-Everything) Network Topology:
                    ┌──────────────────────────────────┐
                    │   Cloud Data Center               │
                    │   (HD map updates, fleet mgmt)    │
                    └──────────────┬───────────────────┘
                                   │ high latency (100–500ms)
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────┴──────┐          ┌──────┴──────┐          ┌──────┴──────┐
   │   RSU-1     │          │   RSU-2     │          │   RSU-3     │
   │ Intersection│          │ Motorway    │          │ Parking Lot │
   │ Edge Server │          │ Edge Server │          │ Edge Server │
   └──────┬──────┘          └──────┬──────┘          └─────────────┘
          │ low latency (5–20ms)   │
   ┌──────┼──────┐          ┌──────┼──────┐
   │      │      │          │      │      │
  🚗Car1 🚗Car2 🚗Car3    🚗Car4 🚗Car5 🚗Car6
  Agent1 Agent2 Agent3    Agent4 Agent5 Agent6

  Each 🚗 = 1 MARL Agent deciding: process locally OR offload to RSU OR send to cloud
  RSU serves nearby vehicles (V2I: Vehicle-to-Infrastructure)
  Vehicles also communicate directly (V2V: Vehicle-to-Vehicle)
```

---

### 📋 ALL OBJECTIVES — Autonomous Vehicle V2X Edge Computing

#### Objective 1: Minimize Safety-Critical Task Latency (Hard Deadline)
- **Goal**: Ensure collision-warning and obstacle-detection tasks complete within strict deadlines
- **Metric**: Deadline miss rate (%) for tasks with deadline ≤ 100ms
- **Real World**: If pedestrian detection takes >100ms, the car cannot brake in time at 60 km/h
- **Difficulty**: ⭐⭐ Easy — simply assign the highest penalty in reward for deadline misses
- **Unique to AV**: Two-tier deadline: soft (path planning, 200ms) vs hard (collision, 50ms)

#### Objective 2: Minimize End-to-End Latency for All Tasks
- **Goal**: Minimize total processing time across all task types (safety-critical + non-critical)
- **Metric**: Average E2E latency = transmission time + RSU queuing delay + processing time
- **Real World**: Lane change must complete before the car enters the target lane
- **Difficulty**: ⭐ Easiest — direct reward signal, single scalar to minimize

#### Objective 3: Minimize Vehicle Computation Energy (Battery / Fuel)
- **Goal**: Reduce energy spent on local computation (saves EV battery range)
- **Metric**: Energy per task = CPU power × local processing time
- **Real World**: A Tesla Model 3's DRIVE computer uses 72W at full load — offloading saves range
- **Sustainability Link**: Directly matches "Sustainable" keyword in thesis title

#### Objective 4: Minimize RSU Energy Consumption (Green Infrastructure)
- **Goal**: Reduce total power used by roadside edge servers
- **Metric**: Total RSU energy = sum(RSU_power × active_duration) across all RSUs
- **Real World**: A city with 500 RSUs at intersections consumes as much power as a small factory
- **Sustainability Link**: Unique to V2X — infrastructure-level sustainability

#### Objective 5: Maximize RSU Utilization (Load Balancing)
- **Goal**: Spread vehicle task load evenly across RSUs; avoid some RSUs being idle while others overflow
- **Metric**: Standard deviation of RSU utilization across the road network (minimize)
- **Real World**: Morning rush hour overloads city-center RSUs while suburban RSUs sit idle
- **MARL Specific**: Vehicles implicitly load-balance without explicit communication

#### Objective 6: Adaptive Task Rerouting When RSU Goes Offline
- **Goal**: When a roadside unit fails or loses connectivity, vehicles seamlessly switch to next RSU or process locally
- **Metric**: Service recovery time (seconds until all vehicles find an alternative)
- **Real World**: RSU loses power in a storm → 30 cars suddenly lose edge compute support
- **Core to Thesis**: "Replanning in Dynamic Environment" — this is the scenario that fits best
- **Difficulty**: ⭐⭐⭐ Moderate — just change server availability flag in simulation

#### Objective 7: V2V Cooperative Offloading (Vehicle-to-Vehicle)
- **Goal**: When a vehicle is out of RSU range, offload tasks to a nearby vehicle acting as relay
- **Metric**: Tasks completed on time when no RSU available (%)
- **Real World**: In a rural highway with no RSUs, a convoy of trucks shares compute resources
- **Difficulty**: ⭐⭐⭐ Moderate — adds peer-to-peer dimension to the offloading decision
- **MARL Contribution**: Agents cooperate directly, not just through shared infrastructure

#### Objective 8: Prioritized Multi-Task Scheduling (Task Queue Management)
- **Goal**: Given multiple pending tasks, decide the best order + destination for each
- **Metric**: Weighted sum of tardiness = Σ(priority_weight × max(0, finish_time − deadline))
- **Real World**: Car receives 5 tasks simultaneously; must prioritize collision warning over HD map update
- **Difficulty**: ⭐⭐ Easy — add priority weighting to existing reward function

---

### 🔬 METHODOLOGIES — Autonomous Vehicle V2X Edge Computing

#### Methodology 1: DQN — Single Vehicle Offloading (Simplest Start)
| Component | Detail |
|---|---|
| **State** | [task_type, task_size, deadline_urgency, vehicle_speed, rsu1_load, rsu2_load, bandwidth] |
| **Action** | Discrete: {0=local, 1=offload_RSU1, 2=offload_RSU2, 3=offload_cloud} |
| **Reward** | −latency − α×energy − β×deadline_miss_penalty |
| **Network** | 2-layer MLP (64→32→4 actions) |
| **Training Steps** | 100,000 steps (fast to train) |
| **Best For** | Single-vehicle baseline; understand the offloading decision first |

**Concrete Example Walkthrough:**
```
Timestep: Car detects pedestrian at 40m, moving at 50 km/h
  State: [task=obstacle_detection, size=2MB, deadline=80ms, speed=50kmh,
          rsu1_load=45%, rsu2_load=80%, bandwidth=50Mbps]

  DQN evaluates Q-values:
    Q(local)     = −0.85  (local processing = 120ms > deadline → too slow)
    Q(RSU-1)     = −0.12  (RSU-1 lightly loaded = 60ms ✓)
    Q(RSU-2)     = −0.45  (RSU-2 heavily loaded = 150ms > deadline)
    Q(cloud)     = −0.95  (cloud WAN = 300ms >> deadline)

  Best action: offload to RSU-1
  Result: 60ms latency, 0.01J energy, deadline met ✓
  Reward: −0.06 − 0.005 + 0 = −0.065
```

---

#### Methodology 2: PPO — Multi-Task Queue per Vehicle
| Component | Detail |
|---|---|
| **State** | Task queue (5 tasks with type/size/deadline) + RSU states + vehicle position |
| **Action** | For each task: (destination, priority_level) — mixed discrete |
| **Reward** | −Σ(w_i × latency_i) − γ×energy + δ×safety_tasks_on_time |
| **Key Advantage** | Handles the realistic case where a car has a backlog of tasks, not just one |
| **Training Steps** | 200,000 steps |
| **Real-World Match** | Accurately models a self-driving car's actual compute pipeline |

---

#### Methodology 3: MAPPO — Fleet of Vehicles as Multi-Agent System (Core Thesis)
| Component | Detail |
|---|---|
| **Agents** | Each vehicle = 1 independent RL agent |
| **Local Observation** | Own task queue + own position + visible RSU loads + V2V neighbor info |
| **Global Critic (training)** | Sees ALL vehicles' states + ALL RSU loads across the road network |
| **Action** | Each vehicle decides where to send its next task |
| **Shared Reward** | −total_fleet_latency − total_RSU_energy − deadline_misses |
| **Individual Reward** | −own_latency − own_energy_consumption |
| **CTDE** | Train centrally (offline simulation), deploy each car independently |

**Why CTDE is Perfect for Autonomous Vehicles:**
```
Training Phase (done at Tesla/Waymo headquarters):
  Simulate 1000 vehicles on a city road network
  Central critic sees: all vehicles, all RSUs, all traffic conditions
  Agents learn: "if RSU ahead is congested at rush hour, start offloading to RSU 2 earlier"

Deployment Phase (runs in each car, forever):
  Each car runs its own policy independently on its DRIVE computer
  No internet connection required for offloading decisions
  Car can make decisions even if 4G/5G network is partially down
  → This is the key safety argument: decentralized resilience
```

**Fleet Load Balancing Without Communication:**
```
Rush hour at city intersection:
  Vehicle A arrives: RSU load = 30% → offloads obstacle detection to RSU
  Vehicle B arrives: RSU load = 55% → still offloads (below threshold)
  Vehicle C arrives: RSU load = 78% → processes locally (learned threshold)
  Vehicle D arrives: RSU load = 85% → looks for RSU-2, finds 40% → offloads there
  Vehicle E arrives: RSU load = 90% → processes locally

Result: No vehicle explicitly told others what it was doing.
The RSU never got overwhelmed. This is emergent MARL coordination.
```

---

#### Methodology 4: Graph Neural Network + MARL (GNN-MARL) — Advanced Variant
| Component | Detail |
|---|---|
| **Why GNN** | Road network = graph (nodes=intersections/RSUs, edges=roads) |
| **Encoding** | GNN encodes road topology → each vehicle gets spatially-aware embedding |
| **Advantage** | Generalizes to new road layouts not seen in training |
| **Complexity** | ⭐⭐⭐⭐ High — requires PyTorch Geometric |
| **Use In Thesis** | Present as advanced variant in Chapter 6 or as future work |

---

#### Methodology 5: Curriculum Learning — Progressive V2X Scenarios
| Stage | Scenario | What Agent Learns |
|---|---|---|
| **Stage 1** | 1 car, 1 RSU, stable network, no failures | Basic local/offload decision |
| **Stage 2** | 5 cars, 2 RSUs, moderate traffic load | Load-aware offloading |
| **Stage 3** | 20 cars, 3 RSUs, rush hour traffic patterns | Congestion prediction + proactive offloading |
| **Stage 4** | 50 cars, 5 RSUs, RSU failure, V2V relay | Full adaptive replanning |
| **Stage 5** | Mixed road types: urban + highway + rural | Domain generalization |
| **Real World** | Mirrors real deployment: testing → pilot city → full rollout | |

---

#### Methodology 6: Heuristic Baselines (for Comparison)
| Heuristic | Rule | AV Context |
|---|---|---|
| **Always Local** | Process all tasks on onboard computer | Overloads DRIVE chip; increases latency |
| **Always Offload** | Always send to nearest RSU | Fails when RSU is overloaded or car is out of range |
| **Nearest RSU** | Offload to geographically closest RSU | Ignores RSU load; causes hotspots |
| **Least Loaded RSU** | Offload to least-loaded RSU in range | Better but ignores transmission distance |
| **Priority-Based Local** | Hard safety tasks = local; soft tasks = edge | Simple but ignores RSU availability |

---

### 🛠️ Simulation Tools — How to Build Without Real Cars

```
Option A (Recommended for Students): Custom Gymnasium Environment
  → Build a simplified city grid (10×10 intersections) in Python
  → Place RSUs at each intersection
  → Generate vehicle tasks using random distributions (Poisson arrival)
  → No external simulator needed
  → Total setup time: 3–5 hours

Option B: SUMO + Custom Gym Wrapper
  → SUMO (Simulation of Urban Mobility) = free, widely used traffic simulator
  → Provides realistic vehicle movement traces
  → Custom wrapper extracts vehicle positions → feeds into offloading RL env
  → Total setup time: 1–2 days
  → pip install traci (SUMO Python interface)

Option C: VeReMi Dataset
  → Real-world Vehicle-to-Vehicle message traces
  → Use for task arrival patterns and message sizes
  → No simulator needed — just replay the trace
```

---

### 📦 Tech Stack — Autonomous Vehicle Edge Computing
```bash
# Core ML/RL
pip install gymnasium stable-baselines3 numpy matplotlib pandas torch

# Traffic simulation (Option B)
# Install SUMO: https://sumo.dlr.de/docs/Installing/index.html
pip install traci eclipse-sumo    # SUMO Python interface

# Graph neural network (Option C, advanced)
pip install torch-geometric       # for GNN-MARL variant
```

### 📁 Datasets & Benchmarks — AV-Specific
| Dataset | Source | What It Provides |
|---|---|---|
| **SUMO Synthetic** | sumo.dlr.de (free) | Realistic vehicle mobility on configurable road maps |
| **VeReMi Dataset** | IEEE open access | Real V2V message traces (size, frequency, content type) |
| **LuST Scenario** | SUMO community | Luxembourg city traffic trace (24 hours, 9,000 vehicles) |
| **InD Dataset** | RWTH Aachen (free) | Real intersection vehicle trajectories (Germany) |
| **Custom Generator** | Your Python script | Fully controllable Poisson vehicle arrivals |

---

### ⚡ Quick-Start Code Skeleton: V2X Autonomous Vehicle Edge (Gymnasium)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Task types with their deadlines (milliseconds, normalized to 0-1 where 1=1000ms)
TASK_TYPES = {
    "collision_warning":     {"size": 0.1, "deadline": 0.02, "priority": 10},
    "obstacle_detection":    {"size": 0.3, "deadline": 0.08, "priority": 8},
    "lane_change_planning":  {"size": 0.2, "deadline": 0.15, "priority": 7},
    "traffic_sign_reading":  {"size": 0.4, "deadline": 0.30, "priority": 5},
    "hd_map_update":         {"size": 0.9, "deadline": 0.80, "priority": 2},
}

class AutonomousVehicleEdgeEnv(gym.Env):
    """
    Autonomous Vehicle V2X Edge Computing Environment.
    Simulates a single self-driving car deciding how to offload compute tasks
    to Roadside Unit (RSU) edge servers or process locally.

    This is the recommended student implementation — no SUMO/hardware needed.
    """
    def __init__(self, n_rsus=2, max_tasks_per_episode=30):
        super().__init__()
        self.n_rsus = n_rsus
        self.max_tasks = max_tasks_per_episode
        self.task_names = list(TASK_TYPES.keys())

        # Actions: 0=local, 1..n_rsus=RSU offload, last=cloud
        self.action_space = spaces.Discrete(1 + n_rsus + 1)

        # State: [task_type (one-hot: 5), task_size, deadline_urgency,
        #         vehicle_speed_norm, rsu1_load, rsu2_load, bandwidth_norm]
        obs_dim = len(self.task_names) + 1 + 1 + 1 + n_rsus + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, seed=None):
        self.tasks_done = 0
        self.rsu_loads = np.random.uniform(0.1, 0.4, self.n_rsus)
        self.vehicle_speed = np.random.uniform(0.3, 1.0)  # 0=stopped, 1=120km/h
        self.bandwidth = np.random.uniform(0.4, 1.0)       # V2I link quality
        return self._new_task_obs(), {}

    def _new_task_obs(self):
        """Sample a new task and build observation vector."""
        self.task_idx = np.random.randint(len(self.task_names))
        task = TASK_TYPES[self.task_names[self.task_idx]]
        self.current_task_size = task["size"]
        self.current_task_deadline = task["deadline"]
        self.current_task_priority = task["priority"]

        one_hot = np.zeros(len(self.task_names))
        one_hot[self.task_idx] = 1.0

        return np.concatenate([
            one_hot,
            [self.current_task_size],
            [self.current_task_deadline],
            [self.vehicle_speed],
            self.rsu_loads,
            [self.bandwidth]
        ]).astype(np.float32)

    def step(self, action):
        latency, energy = self._compute_latency_energy(action)

        # Hard deadline miss — especially penalized for safety-critical tasks
        deadline_penalty = 0.0
        if latency > self.current_task_deadline:
            deadline_penalty = -self.current_task_priority * 2.0

        # Reward: minimize latency + energy; bonus for meeting deadline
        reward = (-latency * self.current_task_priority
                  - 0.3 * energy
                  + deadline_penalty)

        # Dynamics: RSU loads drift over time (simulates other vehicles using them)
        self.rsu_loads = np.clip(
            self.rsu_loads + np.random.uniform(-0.08, 0.12, self.n_rsus),
            0.0, 1.0
        )
        # Speed and bandwidth change (vehicle moves, signal varies)
        self.vehicle_speed = float(np.clip(
            self.vehicle_speed + np.random.uniform(-0.05, 0.05), 0.0, 1.0
        ))
        self.bandwidth = float(np.clip(
            self.bandwidth + np.random.uniform(-0.1, 0.1), 0.1, 1.0
        ))

        self.tasks_done += 1
        done = self.tasks_done >= self.max_tasks
        next_obs = self._new_task_obs() if not done else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        action_names = ["Local"] + [f"RSU-{i+1}" for i in range(self.n_rsus)] + ["Cloud"]
        return next_obs, reward, done, False, {
            "task": self.task_names[self.task_idx],
            "action": action_names[action],
            "latency_ms": round(latency * 1000),
            "deadline_ms": round(self.current_task_deadline * 1000),
            "met_deadline": latency <= self.current_task_deadline,
            "energy": round(energy, 4),
        }

    def _compute_latency_energy(self, action):
        """Compute latency and energy for chosen offloading destination."""
        size = self.current_task_size
        bw = max(self.bandwidth, 0.05)

        if action == 0:
            # Local processing: depends on task size and vehicle speed (CPU throttled when driving fast)
            cpu_load_factor = 1.0 + self.vehicle_speed * 0.5
            latency = size * 0.6 * cpu_load_factor
            energy = size * 0.7
        elif action <= self.n_rsus:
            # Offload to RSU: transmission + RSU queue processing
            rsu_idx = action - 1
            tx_latency = size / bw * 0.1          # 5G-like V2I link
            queue_latency = self.rsu_loads[rsu_idx] * 0.3
            proc_latency = size * 0.05             # RSU is powerful
            latency = tx_latency + queue_latency + proc_latency
            energy = size / bw * 0.15             # transmission energy only
        else:
            # Cloud: long WAN round-trip
            latency = size / bw * 0.1 + 0.35      # 350ms base WAN delay
            energy = size / bw * 0.08             # low device energy
        return float(np.clip(latency, 0, 1)), float(np.clip(energy, 0, 1))


# ── Training ──────────────────────────────────────────────────────────────────
from stable_baselines3 import PPO

env = AutonomousVehicleEdgeEnv(n_rsus=2, max_tasks_per_episode=30)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024,
            batch_size=64, n_epochs=10)
model.learn(total_timesteps=200_000)
model.save("av_edge_offloading_ppo")
print("AV Edge Computing agent training complete!")

# ── Evaluation ────────────────────────────────────────────────────────────────
obs, _ = env.reset()
met, missed = 0, 0
for _ in range(30):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    status = "✅" if info["met_deadline"] else "❌"
    print(f"{status} Task: {info['task']:25s} → {info['action']:6s} | "
          f"Latency: {info['latency_ms']:4d}ms / Deadline: {info['deadline_ms']:4d}ms")
    if info["met_deadline"]:
        met += 1
    else:
        missed += 1
    if done:
        break
print(f"\nDeadline met: {met}/{met+missed} ({100*met/(met+missed):.1f}%)")
```

---

### 🎯 Autonomous Vehicle Thesis Structure (Focused Version)

If you choose to focus your **entire thesis** on the AV/V2X scenario, this is a complete,
publication-ready structure that is achievable in 3–6 months:

```
Chapter 1: Introduction
  → Problem: AV onboard compute is insufficient for all real-time tasks
  → Motivation: 4TB/day sensor data, safety-critical latency requirements
  → Research Gap: Existing offloading algorithms are single-agent, not adaptive to RSU failures
  → Contribution: MARL-based adaptive task offloading for sustainable V2X edge networks

Chapter 2: Background
  2.1 Autonomous vehicle compute pipeline (tasks, latency requirements)
  2.2 V2X architecture (V2I, V2V, MEC at RSU level)
  2.3 Deep RL fundamentals (DQN, PPO, Actor-Critic)
  2.4 Multi-Agent RL (MARL, CTDE paradigm, MAPPO)
  2.5 Related work: MEC offloading papers (2020–2025)

Chapter 3: System Model and Problem Formulation
  3.1 Network model: vehicles, RSUs, cloud tiers
  3.2 Task model: 5 task types with size/deadline/priority
  3.3 Latency model: transmission + queue + processing
  3.4 Energy model: computation energy + transmission energy
  3.5 MARL formulation: state/action/reward/agent definition

Chapter 4: Proposed MARL Framework (MAPPO for V2X)
  4.1 Agent design (each vehicle = 1 PPO agent)
  4.2 CTDE architecture for fleet training
  4.3 Reward function design (multi-objective: latency + energy + deadlines)
  4.4 Curriculum training strategy (Stages 1–5)

Chapter 5: Experiments and Results
  5.1 Simulation setup (road network, RSU placement, traffic patterns)
  5.2 Baseline comparisons (Always-Local, Always-RSU, Least-Loaded, Random)
  5.3 Experiment 1: Deadline miss rate vs number of vehicles (scalability)
  5.4 Experiment 2: Energy savings vs single-agent DQN
  5.5 Experiment 3: RSU failure recovery time (adaptive replanning)
  5.6 Experiment 4: Rush hour vs off-peak performance
  5.7 Ablation: MAPPO vs PPO vs DQN

Chapter 6: Discussion
  → Sustainability impact: energy reduction across vehicle fleet + RSU infrastructure
  → Safety implications: guarantee on hard deadline tasks
  → Scalability analysis: from 5 vehicles to 100 vehicles

Chapter 7: Conclusion
  → Summary of contributions
  → Limitations: simulation vs real 5G network
  → Future work: V2V cooperative offloading, GNN-MARL, hardware deployment
```

---

### 📐 Student Time Estimate — AV Edge Computing Focus

| Phase | Task | Time |
|---|---|---|
| **Week 1** | Read 5 papers on V2X edge computing + MEC | 5–7 hours reading |
| **Week 1** | Build `AutonomousVehicleEdgeEnv` (use skeleton above) | 3–5 hours coding |
| **Week 2** | Train DQN baseline + PPO single vehicle | 2 hours (mostly waiting) |
| **Week 2** | Add RSU failure scenario (flip a load to 1.0) | 1 hour coding |
| **Week 3** | Extend to MAPPO (5 vehicles) | 4–6 hours coding |
| **Week 3–4** | Run experiments, collect results | 3–5 hours |
| **Week 4–5** | Write up results + analysis | Thesis writing phase |
| **Total** | **Full working prototype to thesis-ready results** | **~3–4 weeks** |

---

## 📚 BACKGROUND READING RESOURCES (2–3 Hours Total)

These are the exact resources you need to read before coding. Each link is free to access.
You do NOT need to read all of them deeply — skim headings, read abstract + conclusion, look at figures.

### Tier 1 — Must Read (≈ 90 minutes)

#### 1. What is V2X and Edge Computing for Autonomous Vehicles? (30 min)
> **Read this first — plain English introduction, no math**
- **Wikipedia: V2X (Vehicle-to-Everything)**
  https://en.wikipedia.org/wiki/Vehicle-to-everything
  *What to read*: Overview, V2I section, latency requirements table

- **Wikipedia: Multi-access Edge Computing (MEC)**
  https://en.wikipedia.org/wiki/Mobile_edge_computing
  *What to read*: Introduction, Architecture, Use cases (AV section)

- **ETSI MEC Overview (1-page summary by the standards body)**
  https://www.etsi.org/technologies/multi-access-edge-computing
  *What to read*: The "What is MEC?" summary paragraph only

#### 2. Core Research Paper — Task Offloading in AV Edge Networks (30 min)
> **This is the closest paper to your thesis topic. Read abstract + section III (system model) + results**
- **"Deep Reinforcement Learning for Task Offloading in Mobile Edge Computing"**
  https://arxiv.org/abs/1905.11867
  *What to read*: Abstract, System Model (Section II), reward design (Section III), results table
  *Why it matters*: This is the single-agent DQN baseline you will beat with MAPPO

#### 3. MARL for Edge Computing Survey (30 min)
> **Skim to understand how other researchers set up the MARL problem**
- **"Multi-Agent Reinforcement Learning for Edge Computing: A Survey" (IEEE Access 2022)**
  https://arxiv.org/abs/2205.05038
  *What to read*: Abstract + Table 1 (comparison of existing work) + Section 4.1 (system models)
  *Why it matters*: Shows you what state/action spaces other papers use — validates your design

---

### Tier 2 — Recommended (≈ 60 minutes)

#### 4. PPO Algorithm — Original Paper (15 min skim)
> **You are using PPO in your implementation — read this to understand it**
- **"Proximal Policy Optimization Algorithms" — Schulman et al. (OpenAI, 2017)**
  https://arxiv.org/abs/1707.06347
  *What to read*: Abstract + Section 3 (algorithm) + Figure 2 only
  *Why it matters*: PPO is your core RL algorithm — 1 paragraph in your thesis will cite this

#### 5. MAPPO — Multi-Agent PPO (15 min skim)
> **Your MARL method — the key algorithm that differentiates your thesis**
- **"The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games" (2021)**
  https://arxiv.org/abs/2103.01955
  *What to read*: Abstract + Section 2.1 (MAPPO description) + Table 1 (results)
  *Why it matters*: This is the MARL algorithm your Chapter 4 is built on — cite it

#### 6. Stable-Baselines3 Documentation (15 min)
> **The Python library you will use for training — read the quick-start**
- **Stable-Baselines3 Official Docs**
  https://stable-baselines3.readthedocs.io/en/master/
  *What to read*: "Getting Started" guide + PPO API page
  *Why it matters*: You will call `PPO("MlpPolicy", env, ...)` — know what the parameters mean

#### 7. Gymnasium (OpenAI Gym) Custom Environment Guide (15 min)
> **How to build the `AutonomousVehicleEdgeEnv` class correctly**
- **Gymnasium: Creating Custom Environments**
  https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
  *What to read*: Full tutorial (it is short — 5 minutes to implement, 10 minutes to read)
  *Why it matters*: Your entire simulation is built on this — understand `reset()`, `step()`, `observation_space`

---

### Tier 3 — Optional / Reference Only (bookmark, don't read now)

| Resource | Link | When to Use |
|---|---|---|
| **SUMO Traffic Simulator Docs** | https://sumo.dlr.de/docs/ | If you add realistic vehicle movement (Week 3+) |
| **VeReMi Dataset (V2V message traces)** | https://veremi-dataset.github.io/ | If you want real task arrival patterns |
| **PyTorch Tutorials (if needed)** | https://pytorch.org/tutorials/ | If Stable-Baselines3 is not enough |
| **RLlib Multi-Agent Docs** | https://docs.ray.io/en/latest/rllib/index.html | Alternative to SB3 for true multi-agent training |
| **"Multi-Access Edge Computing: A Survey" (IEEE, 2017)** | https://arxiv.org/abs/1709.01656 | Deep dive into MEC theory for literature review |
| **ETSI MEC standards (technical specs)** | https://www.etsi.org/standards#page=1&search=MEC | Chapter 2 literature review citations |

---

## 🗓️ DETAILED 4-WEEK PLAN: From Scratch to Thesis-Ready

This plan assumes you work **3–4 hours per day**. Each day has a clear deliverable.
**Total estimated hours: 60–80 hours** across 4 weeks.

---

### 📅 WEEK 1 — Foundation: Background Reading + Environment Setup

**Goal by end of Week 1:** A working simulation environment + understand the problem deeply enough to explain it to your supervisor.

---

#### Day 1 (Monday) — Background Reading Session
**Time required: 3 hours**
```
Hour 1: V2X fundamentals
  ✅ Read Wikipedia V2X article (focus on V2I section)
     → https://en.wikipedia.org/wiki/Vehicle-to-everything
  ✅ Read Wikipedia MEC article
     → https://en.wikipedia.org/wiki/Mobile_edge_computing
  ✅ Draw a simple diagram: vehicle → RSU → cloud (on paper)
  ✅ Write down: What are 3 tasks an AV generates? What is their deadline?

Hour 2: Core paper reading
  ✅ Open: https://arxiv.org/abs/1905.11867
  ✅ Read: Abstract (5 min) + System Model Figure (5 min) + Results Table (5 min)
  ✅ Answer: What state did they use? What action? What reward?
  ✅ Write 3 bullet points: "My thesis will improve on this paper by..."

Hour 3: MARL paper skim
  ✅ Open: https://arxiv.org/abs/2205.05038
  ✅ Jump to Table 1 — look at how other papers define agents, state, action
  ✅ Open: https://arxiv.org/abs/2103.01955 (MAPPO paper)
  ✅ Read abstract + Section 2.1 (2 pages only)
  ✅ Write: "My agents = vehicles. My MARL method = MAPPO. Because..."
```
**Day 1 Deliverable:** 1-page handwritten notes with your problem formulation.

---

#### Day 2 (Tuesday) — Environment Setup
**Time required: 3–4 hours**
```
Hour 1: Python environment setup
  ✅ Install Python 3.10+ (if not already installed)
     → https://www.python.org/downloads/
  ✅ Create virtual environment:
     python -m venv av_thesis_env
     source av_thesis_env/bin/activate     # Linux/Mac
     av_thesis_env\Scripts\activate        # Windows
  ✅ Install dependencies:
     pip install gymnasium stable-baselines3 numpy matplotlib pandas torch

Hour 2: Read Gymnasium tutorial
  ✅ Open: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
  ✅ Code the example from the tutorial (not your AV env yet — just get familiar)
  ✅ Run it: python my_first_env.py

Hour 3–4: Build AutonomousVehicleEdgeEnv
  ✅ Copy the code skeleton from this document (Section: "Quick-Start Code Skeleton")
  ✅ Save as: av_edge_env.py
  ✅ Run: python av_edge_env.py
  ✅ Verify: environment resets, steps without error, observations have correct shape
  ✅ Print 5 random actions to confirm the action/observation cycle works
```
**Day 2 Deliverable:** `av_edge_env.py` runs without errors. You can call `env.reset()` and `env.step(action)`.

---

#### Day 3 (Wednesday) — First Training Run: DQN Baseline
**Time required: 3 hours**
```
Hour 1: Train a random agent (sanity check)
  ✅ Write: av_random_baseline.py
     obs, _ = env.reset()
     for _ in range(300):
         action = env.action_space.sample()    # random
         obs, reward, done, _, info = env.step(action)
         print(info)
         if done: obs, _ = env.reset()
  ✅ Run it. Record average deadline miss rate.
  ✅ This is your "Random" baseline — save the number.

Hour 2: Train DQN (simplest RL agent)
  ✅ Write: av_dqn_train.py
     from stable_baselines3 import DQN
     model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=10000)
     model.learn(total_timesteps=100_000)
     model.save("av_dqn_baseline")
  ✅ Run training (takes ~5–10 minutes on a laptop)
  ✅ Watch the reward increase in the output

Hour 3: Evaluate DQN
  ✅ Write: av_dqn_eval.py
     model = DQN.load("av_dqn_baseline")
     obs, _ = env.reset()
     met, total = 0, 0
     for _ in range(100):
         action, _ = model.predict(obs, deterministic=True)
         obs, r, done, _, info = env.step(action)
         met += info["met_deadline"]
         total += 1
         if done: obs, _ = env.reset()
     print(f"DQN Deadline Met: {met/total*100:.1f}%")
  ✅ Record: DQN deadline met rate. Compare to Random.
  ✅ Expected result: DQN should beat random by 10–25%
```
**Day 3 Deliverable:** First result table row: `Random: XX% | DQN: XX%` deadline met rate.

---

#### Day 4 (Thursday) — PPO Training + Comparison
**Time required: 3 hours**
```
Hour 1: Train PPO (your main single-agent method)
  ✅ Write: av_ppo_train.py
     from stable_baselines3 import PPO
     model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4,
                 n_steps=1024, batch_size=64, n_epochs=10)
     model.learn(total_timesteps=200_000)
     model.save("av_ppo_single_agent")
  ✅ Run training (takes ~10–20 minutes on a laptop)

Hour 2: Evaluate PPO + compare with DQN
  ✅ Run same evaluation loop for PPO
  ✅ Record: PPO deadline met rate
  ✅ Compare: Random vs DQN vs PPO
  ✅ Expected: PPO > DQN > Random (if not, check reward function sign)

Hour 3: Plot learning curve
  ✅ Add to training script:
     from stable_baselines3.common.callbacks import EvalCallback
     from stable_baselines3.common.monitor import Monitor
     import matplotlib.pyplot as plt
  ✅ Plot reward over training steps (this becomes Figure 1 in your thesis)
  ✅ Save plot as: results/ppo_learning_curve.png
```
**Day 4 Deliverable:** `results/` folder with first comparison table + learning curve plot.

---

#### Day 5 (Friday) — Add Disruption: RSU Failure Scenario
**Time required: 2–3 hours**
```
Hour 1: Add RSU failure to the environment
  ✅ Open av_edge_env.py
  ✅ Add to reset(): self.rsu_failure_step = np.random.randint(10, 25)
  ✅ Add to step(): 
     if self.tasks_done == self.rsu_failure_step:
         self.rsu_loads[0] = 1.0    # RSU-1 goes offline (fully loaded)
  ✅ Re-run random + DQN + PPO baselines on this harder environment
  ✅ Record: how much does deadline miss rate go up when RSU fails?

Hour 2–3: Retrain PPO on failure environment
  ✅ Retrain model.learn(total_timesteps=200_000) with failure scenario active
  ✅ Compare: PPO (no failure training) vs PPO (trained with failure)
  ✅ Expected: PPO trained WITH failures learns to switch to RSU-2 automatically
  ✅ This is Experiment 3 in your thesis: adaptive replanning under RSU failure
```
**Day 5 Deliverable:** Evidence that your RL agent adapts to RSU failure (core thesis contribution).

---

### 📅 WEEK 2 — Multi-Agent: MAPPO Fleet of Vehicles

**Goal by end of Week 2:** A working 5-vehicle MARL system trained with MAPPO.

---

#### Day 6 (Monday) — Multi-Agent Environment Setup
**Time required: 4 hours**
```
Hours 1–2: Understand MARL library options
  ✅ Read RLlib MARL docs (30 min skim):
     https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-envs
  ✅ OR use the simpler approach: run N copies of env in parallel with shared reward
  ✅ Recommended for students: Use PettingZoo-style wrapper (easier)
     pip install pettingzoo

Hours 3–4: Build multi-vehicle environment
  ✅ Create: av_multi_agent_env.py
  ✅ Wrap AutonomousVehicleEdgeEnv so N vehicles share the same RSU loads
  ✅ Key change: RSU loads increase when any vehicle offloads to them
     (vehicles compete for the same RSU resources — this is the core MARL interaction)
  ✅ Start with N=3 vehicles, RSU capacity=1.0 shared
  ✅ Test: run 3 random agents, confirm RSU loads increase when all 3 offload together
```
**Day 6 Deliverable:** `av_multi_agent_env.py` — 3 vehicles competing for shared RSUs.

---

#### Day 7 (Tuesday) — Train MAPPO (Core Thesis Algorithm)
**Time required: 4 hours**
```
Hours 1–2: Train independent PPO per vehicle (simpler MARL baseline)
  ✅ Train each vehicle with its own PPO model, sharing RSU state
  ✅ This is "Independent PPO" — a common MARL baseline
  ✅ Record: does load balancing emerge? (check if all 3 vehicles don't offload to RSU-1 simultaneously)

Hours 3–4: Train MAPPO
  ✅ Use RLlib MAPPO OR implement CTDE manually:
     → Shared critic: takes concatenated state of all vehicles
     → Individual actors: each vehicle has its own policy network
  ✅ Alternative (simpler): use parameter sharing (all agents use same neural network weights)
     from stable_baselines3 import PPO
     # Train with shared policy — each vehicle observation is independent input
     # This approximates MAPPO for a student implementation
  ✅ Run 200,000 steps
  ✅ Record: fleet average deadline met rate
```
**Day 7 Deliverable:** MAPPO (or shared-policy PPO) trained on 3-vehicle fleet.

---

#### Day 8 (Wednesday) — Scale Up: 5 Vehicles + Rush Hour Simulation
**Time required: 3 hours**
```
Hours 1–2: Increase to 5 vehicles
  ✅ Change n_vehicles = 5 in your multi-agent env
  ✅ Retrain (takes longer — 30–40 min on laptop)
  ✅ Record: does performance degrade as more vehicles compete for RSUs?

Hour 3: Add rush hour traffic pattern
  ✅ Add to environment: self.rush_hour = (20 <= self.tasks_done <= 40)
  ✅ During rush hour: new_task_arrival_rate *= 2 (double the task generation rate)
  ✅ Run both: PPO single-agent vs MAPPO fleet during rush hour
  ✅ Record: which handles rush hour better?
  ✅ Expected: MAPPO distributes load; single agent gets overwhelmed
```
**Day 8 Deliverable:** Experiment 4 results: Rush hour vs off-peak performance comparison.

---

#### Day 9 (Thursday) — Results Table + Visualizations
**Time required: 3 hours**
```
Hours 1–2: Collect all results into a table
  ✅ Create: results/experiment_results.csv
  
  Method              | Deadline Met% | Avg Latency(ms) | Energy | RSU Utilization Std
  Random              |               |                 |        |
  Always-Local        |               |                 |        |
  Always-RSU          |               |                 |        |
  Least-Loaded        |               |                 |        |
  DQN (single agent)  |               |                 |        |
  PPO (single agent)  |               |                 |        |
  MAPPO (5 vehicles)  |               |                 |        |

  ✅ Run each method for 1000 steps, compute averages
  ✅ This is Table 1 in your thesis Chapter 5

Hour 3: Generate plots
  ✅ Plot 1: Learning curve (reward vs training steps) — ppo vs mappo
  ✅ Plot 2: Bar chart — deadline met rate for all methods
  ✅ Plot 3: Line chart — RSU utilization over time (rush hour visible as spike)
  ✅ Save all to results/ folder
```
**Day 9 Deliverable:** `results/` folder with all plots + results CSV. Core of Chapter 5.

---

#### Day 10 (Friday) — Buffer Day + Review
**Time required: 2 hours**
```
✅ Re-read your Day 1 notes
✅ Check: does your MAPPO result beat all baselines?
✅ If PPO beats MAPPO: increase n_vehicles (MARL benefit shows at higher contention)
✅ If results look wrong: check reward function — energy coefficient might be too high
✅ Commit all code to GitHub
✅ Backup results folder
```

---

### 📅 WEEK 3 — Experiments: Ablation Studies + Advanced Scenarios

**Goal by end of Week 3:** All 5 experiments complete, ready to write.

---

#### Day 11 (Monday) — Experiment: Scalability Test
**Time required: 3–4 hours**
```
✅ Run your MAPPO with: n_vehicles = 1, 5, 10, 20, 30
✅ Record deadline met rate for each fleet size
✅ Plot: x-axis = number of vehicles, y-axis = deadline met rate
✅ Expected: MAPPO degrades gracefully; Always-RSU collapses when all offload simultaneously
✅ This becomes Figure 3 in your thesis (scalability analysis)
```

---

#### Day 12 (Tuesday) — Experiment: Sustainability Analysis
**Time required: 3 hours**
```
✅ Add energy tracking to your eval loop:
   total_vehicle_energy = sum(info["energy"] for all vehicles)
   total_rsu_energy_proxy = sum(rsu_loads) * 0.1    # simple linear model
✅ Run all methods, record energy
✅ Calculate: how much energy does MAPPO save vs Always-Local?
✅ This is your "Sustainable" contribution → maps directly to thesis title keyword
```

---

#### Day 13 (Wednesday) — Experiment: RSU Failure Recovery
**Time required: 3 hours**
```
✅ Simulate: at step 50, RSU-1 goes offline (load = 1.0, stays there)
✅ Measure: how many steps until performance recovers to pre-failure level?
✅ Compare: PPO (single agent) vs MAPPO fleet recovery speed
✅ This is Experiment 3: Adaptive Replanning (core of "Replanning" in thesis title)
✅ Plot: performance over time with failure event marked with vertical line
```

---

#### Day 14 (Thursday) — Ablation Study
**Time required: 3 hours**
```
✅ Remove components one by one to show each contributes:
   Ablation A: Remove deadline penalty from reward → deadline met rate drops
   Ablation B: Remove energy term from reward → energy increases
   Ablation C: Single agent vs multi-agent → load balancing disappears
   Ablation D: No curriculum learning vs curriculum → training stability
✅ Record results for each ablation
✅ This shows your design choices were intentional and correct
```

---

#### Day 15 (Friday) — Buffer + Code Cleanup
**Time required: 2 hours**
```
✅ Clean up all Python files (add comments, remove debug prints)
✅ Create requirements.txt: pip freeze > requirements.txt
✅ Write a README.md for your code repository
✅ Organize: src/ (env code), experiments/ (training scripts), results/ (plots + CSV)
✅ Commit to GitHub with clear commit messages
```

---

### 📅 WEEK 4 — Writing: Thesis Draft

**Goal by end of Week 4:** Full thesis draft (Chapter 1 + 3 + 5 completed, others outlined).

---

#### Day 16 (Monday) — Write Chapter 3: System Model
**Time required: 4–5 hours**
```
✅ Section 3.1: Network model (copy your architecture diagram into LaTeX/Word)
✅ Section 3.2: Task model (table of 5 task types, size, deadline, priority)
✅ Section 3.3: Latency model (write the equations from your _compute_latency_energy function)
✅ Section 3.4: Energy model (same)
✅ Section 3.5: MARL formulation
   - State space S = (write out the observation vector from your code)
   - Action space A = {local, RSU-1, RSU-2, cloud}
   - Reward function R = (write out your reward equation)
   - Agent definition: each vehicle i ∈ {1,...,N} is an independent RL agent

Tools:
   LaTeX template: https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn
   Free alternative: Google Docs with equation editor
```
**Day 16 Deliverable:** Chapter 3 first draft (3–5 pages).

---

#### Day 17 (Tuesday) — Write Chapter 5: Experiments
**Time required: 4–5 hours**
```
✅ Section 5.1: Simulation setup
   - Copy your environment parameters (n_vehicles=5, n_rsus=2, max_tasks=30)
   - "We implemented a custom Gymnasium environment in Python 3.10"
   - "Training used Stable-Baselines3 v2.x with PPO/MAPPO"

✅ Section 5.2: Baselines (describe each heuristic in 1 sentence)

✅ Section 5.3–5.7: One section per experiment
   - For each: describe setup (2 sentences) + insert your plot/table + write 3 observation sentences
   - "As shown in Figure X, MAPPO achieves Y% deadline compliance, outperforming the DQN baseline by Z%"

✅ Insert all plots from your results/ folder
✅ Insert results table (Table 1 from your CSV)
```
**Day 17 Deliverable:** Chapter 5 first draft (6–10 pages).

---

#### Day 18 (Wednesday) — Write Chapter 4: Proposed Method
**Time required: 4 hours**
```
✅ Section 4.1: Agent design
   - Copy your AutonomousVehicleEdgeEnv observation space description
   - Explain: "Each vehicle maintains a local policy network π_θ: S → A"

✅ Section 4.2: CTDE architecture
   - Draw: centralized training box → N vehicle actors box
   - Explain difference between training (central critic) and deployment (local only)

✅ Section 4.3: Reward function
   - Write mathematical equation for your reward
   - Justify each coefficient (why α=0.3 for energy? experiment with ablation)

✅ Section 4.4: Curriculum training
   - Describe your 5-stage curriculum (from the earlier section of this document)
```
**Day 18 Deliverable:** Chapter 4 first draft (4–6 pages).

---

#### Day 19 (Thursday) — Write Chapter 1: Introduction
**Time required: 3–4 hours**
```
✅ Paragraph 1: The problem (4–5 lines)
   "Autonomous vehicles generate up to 4TB of sensor data per day [cite].
    Onboard processing alone is insufficient for safety-critical real-time tasks [cite].
    Mobile edge computing (MEC) offers low-latency compute at roadside infrastructure..."

✅ Paragraph 2: The gap (3–4 lines)
   "Existing offloading algorithms assume single-agent decision making [cite: arxiv 1905.11867].
    They fail to account for multi-vehicle contention for shared RSU resources..."

✅ Paragraph 3: Your contribution (4 bullet points)
   1. A multi-agent simulation environment for V2X task offloading
   2. MAPPO-based fleet coordination for adaptive RSU offloading
   3. Demonstrated RSU failure recovery through learned replanning
   4. Energy-efficient scheduling that reduces fleet + RSU energy by X%

✅ Paragraph 4: Paper structure (1 sentence per chapter)

Citations to use:
   [1] DRL offloading: https://arxiv.org/abs/1905.11867
   [2] MAPPO: https://arxiv.org/abs/2103.01955
   [3] MEC survey: https://arxiv.org/abs/1709.01656
   [4] MARL for edge: https://arxiv.org/abs/2205.05038
```
**Day 19 Deliverable:** Chapter 1 complete (2–3 pages).

---

#### Day 20 (Friday) — Write Abstract + Conclusion + Polish
**Time required: 3 hours**
```
Hour 1: Write Abstract (250 words)
  Structure: Problem (2 sentences) + Gap (1 sentence) + Method (2 sentences) +
             Key Results (2 sentences) + Conclusion (1 sentence)
  Tip: Write abstract LAST so all numbers are correct

Hour 2: Write Chapter 7: Conclusion
  ✅ Section 7.1: Summary of contributions (4 bullet points from Chapter 1 + results)
  ✅ Section 7.2: Limitations (2–3 bullets: simulation-only, simplified channel model, etc.)
  ✅ Section 7.3: Future work
    - V2V cooperative offloading (Objective 7 from this document)
    - GNN-MARL for scalable road network topology
    - Real 5G testbed deployment

Hour 3: Final polish
  ✅ Add references section (use Google Scholar to get BibTeX for each paper)
  ✅ Check: all figures labeled, all tables numbered, all equations numbered
  ✅ Read abstract + conclusion together — do they tell the same story?
  ✅ Submit draft to supervisor
```
**Day 20 Deliverable:** 🎓 Complete thesis first draft ready for supervisor feedback.

---

## 📋 COMPLETE RESOURCE CHECKLIST

### Papers to Cite (All Free)
| # | Paper | Link | Chapter Used In |
|---|---|---|---|
| 1 | DRL for Task Offloading (MEC) | https://arxiv.org/abs/1905.11867 | Ch 1, 2, 5 (baseline) |
| 2 | MAPPO Paper | https://arxiv.org/abs/2103.01955 | Ch 4 (method) |
| 3 | MARL for Edge Computing Survey | https://arxiv.org/abs/2205.05038 | Ch 2 (lit review) |
| 4 | MEC Survey (IEEE 2017) | https://arxiv.org/abs/1709.01656 | Ch 2 (background) |
| 5 | PPO Algorithm (OpenAI) | https://arxiv.org/abs/1707.06347 | Ch 4 (method) |
| 6 | DQN Paper (DeepMind) | https://arxiv.org/abs/1312.5602 | Ch 2 (baseline) |
| 7 | CTDE for MARL (MADDPG) | https://arxiv.org/abs/1706.02275 | Ch 4 (architecture) |

### Tools + Docs
| Tool | Link | Purpose |
|---|---|---|
| Stable-Baselines3 | https://stable-baselines3.readthedocs.io | PPO/DQN training |
| Gymnasium | https://gymnasium.farama.org | Custom env base class |
| Matplotlib | https://matplotlib.org/stable/gallery | Plotting results |
| Overleaf (LaTeX) | https://www.overleaf.com | Thesis writing (free tier) |
| Google Scholar | https://scholar.google.com | Get BibTeX citations |
| arXiv | https://arxiv.org | Find papers (all free) |

---

## 📦 Final Folder Structure (After 4 Weeks)
```
av_v2x_thesis/
├── src/
│   ├── av_edge_env.py            # Single-vehicle Gymnasium environment
│   └── av_multi_agent_env.py     # Multi-vehicle MARL environment
├── experiments/
│   ├── av_random_baseline.py
│   ├── av_dqn_train.py
│   ├── av_ppo_single_agent.py
│   ├── av_mappo_fleet.py
│   ├── av_eval_all_methods.py
│   └── av_ablation_study.py
├── results/
│   ├── experiment_results.csv    # All numbers for Table 1
│   ├── ppo_learning_curve.png
│   ├── deadline_comparison_bar.png
│   ├── rsu_utilization_time.png
│   └── scalability_fleet_size.png
├── thesis/
│   ├── chapter1_introduction.tex (or .docx)
│   ├── chapter3_system_model.tex
│   ├── chapter4_methodology.tex
│   └── chapter5_experiments.tex
├── models/
│   ├── av_dqn_baseline.zip
│   ├── av_ppo_single_agent.zip
│   └── av_mappo_fleet.zip
├── requirements.txt
└── README.md
```

---

*Generated for: MULTI-AGENT DEEP REINFORCEMENT LEARNING FOR SUSTAINABLE ADAPTIVE SCHEDULING AND REPLANNING IN DYNAMIC ENVIRONMENTS*
*Domains covered: Project Management (Top 1) | Cloud Computing (Top 2) | Edge Computing (Top 3) | AV/V2X Specialization*
