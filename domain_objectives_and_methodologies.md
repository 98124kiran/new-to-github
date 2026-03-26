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

*Generated for: MULTI-AGENT DEEP REINFORCEMENT LEARNING FOR SUSTAINABLE ADAPTIVE SCHEDULING AND REPLANNING IN DYNAMIC ENVIRONMENTS*
*Domains covered: Project Management (Top 1) | Cloud Computing (Top 2) | Edge Computing (Top 3)*
