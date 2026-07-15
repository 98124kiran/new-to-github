MULTI-AGENT DEEP REINFORCEMENT LEARNING FOR SUSTAINABLE ADAPTIVE SCHEDULING AND REPLANNING IN DYNAMIC ENVORNMENT

## Event-Driven Replanning Framework for Real-Time Disturbance Handling

### 1) Disturbance Event Model

The framework defines five disturbance categories:
- **MachineFailure**
- **QualityDrift**
- **MaterialDelay**
- **UrgentOrder**
- **EnergyConstraint**

Each event includes:
- `eventId`
- `eventType`
- `source`
- `timestamp`
- `severity` (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)
- `priority` (numeric, where lower value means higher priority)
- `context` (machine/job/line/order details)

Priority rules:
- `CRITICAL` events always preempt lower-severity events.
- Safety and compliance events are always prioritized over throughput-only events.
- If severity is equal, earlier timestamp wins.
- If severity and timestamp are equal, impact breadth (number of affected resources) decides.

### 2) Event-Driven Monitoring Layer

The monitoring layer continuously subscribes to shop-floor telemetry and system logs. It performs:
- **Validation**: schema, required fields, timestamp sanity, duplicate detection.
- **Classification**: map raw signals to event categories.
- **Enrichment**: attach line, job, due-date, and asset metadata.

Processing stages:
1. Ingest event stream
2. Validate and normalize payload
3. Classify to disturbance type
4. Compute severity and priority
5. Push to impact-assessment queue

### 3) Impact Assessment Before Replanning

Before any replanning action, impact assessment determines:
- affected jobs and work orders
- affected machines, lines, operators, and materials
- due-date risk and SLA risk
- KPI impact (throughput, tardiness, quality, downtime, energy)

Assessment outputs:
- `impactScope` (`LOCAL`, `LINE`, `PLANT`)
- `riskScore` (0-100)
- `replanningRequired` (true/false)
- recommended response mode

### 4) Tiered Replanning Logic

Replanning follows progressive escalation:

1. **Local Repair (Machine/Cell)**  
   Adjust queue, reroute within cell, or temporary speed/sequence adjustment.

2. **Line-Level Replanning**  
   Rebalance tasks across machines in a line when local repair is insufficient.

3. **Plant-Level Replanning**  
   Re-optimize across lines only when disturbance spreads beyond one line or affects global commitments.

Escalation occurs only when lower tier cannot meet constraints.

### 5) Multi-Agent Coordination

The framework coordinates four agents:
- **SchedulingAgent**
- **MaintenanceAgent**
- **QualityAgent**
- **LogisticsAgent**

Coordination protocol:
1. Share event context and impact summary
2. Each agent proposes feasible action set
3. Resolve conflicts using priority rules and hard constraints
4. Publish a joint action plan with execution order

Conflict-free guarantees:
- Safety constraints are hard constraints
- Due-date and quality constraints are jointly validated
- No plan is executed until all mandatory constraints pass

### 6) Stability Controls

To avoid excessive plan churn, the framework enforces:
- **Replanning thresholds**: minimum risk score and minimum KPI impact before triggering replanning
- **Cooldown windows**: suppress repeated replanning for near-identical events within a short interval
- **Change penalties**: penalize frequent schedule reshuffles and unnecessary resource switching

This ensures responsiveness without unstable oscillation.

### 7) Fallback and Recovery Paths

For critical disturbances:
- **Safe mode**: switch affected units to safe operational profile
- **Manual checkpoint**: require operator approval for high-risk overrides
- **Rollback**: revert to last feasible validated plan if new plan execution fails
- **Recovery verification**: confirm machine health and schedule consistency before exiting fallback mode

### 8) Real-Time KPI Tracking

The framework tracks these KPIs continuously:
- Event response latency
- Replanning computation time
- Schedule stability index
- On-time delivery recovery rate
- Downtime reduction

KPI pipeline:
1. Collect execution telemetry
2. Compute KPI deltas per event and per shift
3. Compare against baseline
4. Trigger tuning actions if KPI degradation crosses thresholds

## Expected Outcome

This implementation enables fast and structured disturbance handling with controlled replanning escalation, coordinated multi-agent decisions, and measurable performance improvements in dynamic manufacturing environments.
