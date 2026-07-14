"""Divergence-pilot probe set — one probe per divergent class (PC-01..PC-10).

Each probe is run twice: once with its CORRECT-frame forcing scaffold and once
with its FOIL-frame forcing scaffold, both placed at the TOP of the branch
frame (the only place that conditions the trajectory on a no-KV-cache API).

The forcing strings name the APPROACH, never the answer, and are kept parallel
in structure across correct/foil so neither is privileged by phrasing. Ground
truth + foils are carried in `rule`; they are the embedded-ground-truth payload
the standing decorrelated-recompute gate checks before any live run.

Prompts end with the harness-appended answer-line instruction (see pilot.py),
so the bodies here omit it to keep the forcing comparison clean.
"""
from __future__ import annotations

PROBES = [
    {
        "id": "PC-01",
        "prompt": (
            "A read request fans out in parallel to 10 replicas and completes only when "
            "ALL 10 replies have arrived. Each replica's response latency is independent and "
            "uniformly distributed between 100 ms and 200 ms. What is the expected end-to-end "
            "completion time of the read request, in milliseconds, rounded to the nearest millisecond?"
        ),
        "correct": "the quantity asked for is an order statistic — the expected value of the SLOWEST of the parallel draws, computed from the distribution of that extreme, not an average of one component",
        "foil": "the quantity asked for is a central-tendency value — the average latency of a single component's distribution",
        "rule": {"kind": "numeric", "lo": 189, "hi": 193, "gt": 191, "foils": [150, 200]},
    },
    {
        "id": "PC-02",
        "prompt": (
            "An anomaly alert for a payment service fires on 99% of days that contain a real outage. "
            "On days with no outage, it still fires 2% of the time. Real outages occur on 1 day in 500. "
            "The alert just fired today. What is the probability that there is a real outage today, "
            "expressed as a percentage?"
        ),
        "correct": "combine the given forward rates with the base rate to get the REVERSE conditional — the probability of the condition given that the detector fired",
        "foil": "read the answer directly off one of the conditional rates stated in the problem",
        "rule": {"kind": "numeric", "lo": 8.5, "hi": 9.6, "gt": 9.0, "foils": [99, 98, 2]},
    },
    {
        "id": "PC-03",
        "prompt": (
            "A data migration copies 8,000 MB. The first 4,000 MB transfer at a sustained 100 MB/s; "
            "the remaining 4,000 MB transfer at a sustained 25 MB/s. What is the overall average "
            "throughput of the migration, in MB/s?"
        ),
        "correct": "compute the overall rate as total quantity divided by total time, accumulated across the two segments",
        "foil": "compute the overall rate as the average of the two per-segment rates",
        "rule": {"kind": "numeric", "lo": 39.5, "hi": 40.5, "gt": 40, "foils": [62.5]},
    },
    {
        "id": "PC-04",
        "prompt": (
            "A router assigns each incoming request to one of 12 shards. You cannot predict or "
            "influence which shard any given request goes to. What is the minimum number of requests "
            "that GUARANTEES, under every possible assignment of requests to shards, that at least one "
            "shard has received at least 3 requests?"
        ),
        "correct": "this is a worst-case guarantee — find the count that forces the outcome under the most adversarial possible placement (pigeonhole)",
        "foil": "this is a probabilistic / expected-occupancy question — find the count at which the outcome becomes likely",
        "rule": {"kind": "numeric", "lo": 25, "hi": 25, "gt": 25, "foils": []},
    },
    {
        "id": "PC-05",
        "prompt": (
            "Five independent build tasks take 8, 7, 6, 5, and 4 hours. You have 3 identical build "
            "agents, all available from time zero. Each task runs on exactly one agent and cannot be "
            "split or paused; the agents work in parallel. What is the minimum possible time, in hours, "
            "until ALL tasks are finished?"
        ),
        "correct": "this is a scheduling problem with indivisible tasks — find the best assignment of tasks to agents; the answer is the makespan, which can exceed total-work divided by agent count",
        "foil": "the answer is total work divided by the number of agents",
        "rule": {"kind": "numeric", "lo": 11, "hi": 11, "gt": 11, "foils": [10]},
    },
    {
        "id": "PC-06",
        "prompt": (
            "A log volume starts empty and grows continuously at 4 GB per hour. A compaction job runs "
            "every 6 hours (at t = 6 h, 12 h, 18 h, ...) and instantly deletes 12 GB; in this scenario "
            "at least 12 GB is always present whenever it runs. The volume's capacity is 40 GB. At what "
            "time t, in hours from the start, does usage FIRST reach 40 GB?"
        ),
        "correct": "step the stock through each discrete event over time; the threshold is first crossed at a peak between compaction events",
        "foil": "collapse the dynamics to one net rate per cycle and extrapolate linearly to the threshold",
        "rule": {"kind": "numeric", "lo": 15.9, "hi": 16.1, "gt": 16, "foils": [20, 10]},
    },
    {
        "id": "PC-07",
        "prompt": (
            "Two build pipelines ran last quarter. Atlas: frontend repos 81 successes out of 87 builds; "
            "backend repos 192 successes out of 263 builds. Borealis: frontend repos 234 successes out of "
            "270 builds; backend repos 55 successes out of 80 builds. Your team builds ONLY frontend repos "
            "and will route all of its builds to one pipeline. Based on these counts, which pipeline gives "
            "your team's builds the higher success probability? Choose Atlas or Borealis."
        ),
        "correct": "compare the pipelines only within the specific subpopulation the question fixes — the relevant stratum",
        "foil": "compare the pipelines on their pooled totals across all subpopulations combined",
        "rule": {"kind": "token", "correct": "Atlas", "menu": ["Atlas", "Borealis"], "foils": ["Borealis"]},
    },
    {
        "id": "PC-08",
        "prompt": (
            "A web stack has exactly four components: a load balancer, an API service, a cache, and a "
            "database. Read requests flow: load balancer -> API service -> cache -> database. Write "
            "requests flow: load balancer -> API service -> database (no cache hop). Reads sent with the "
            "header X-Bypass-Cache flow: load balancer -> API service -> database. Observations since "
            "14:00 today: (1) all writes succeed; (2) all normal reads time out; (3) all reads sent with "
            "X-Bypass-Cache succeed; (4) the cache's own health endpoint reports OK; (5) the API service "
            "was redeployed at 13:35. Exactly one component is faulty. Which one? Choose load balancer, "
            "API service, cache, or database."
        ),
        "correct": "isolate the fault by elimination over which flows pass and fail — a component on any passing flow is exonerated; the unique remaining one is the fault",
        "foil": "attribute the fault to the most recently changed component",
        "rule": {"kind": "token", "correct": "cache", "menu": ["load balancer", "API service", "cache", "database"], "foils": ["API service"]},
    },
    {
        "id": "PC-09",
        "prompt": (
            "A courier depot will be placed on a straight service road. Five customer sites sit at mile "
            "markers 2, 5, 11, 19, and 23. Each day the courier makes one separate round trip from the "
            "depot to EACH of the five sites. At which mile marker should the depot be placed to MINIMIZE "
            "THE TOTAL distance driven per day?"
        ),
        "correct": "place to minimize the SUM of distances over all sites — the median location",
        "foil": "place to minimize the MAXIMUM distance to any one site (the midrange), or at the mean position",
        "rule": {"kind": "numeric", "lo": 10.9, "hi": 11.1, "gt": 11, "foils": [12.5, 12]},
    },
    {
        "id": "PC-10",
        "prompt": (
            "A nightly batch can run under one of two strategies. Strategy Fixed always completes in "
            "exactly 60 minutes. Strategy Turbo completes in 30 minutes with probability 0.7, and in 150 "
            "minutes with probability 0.3. Which strategy has the LOWER EXPECTED completion time? Choose "
            "Fixed or Turbo."
        ),
        "correct": "decide by comparing expected values — the probability-weighted average completion time of each strategy",
        "foil": "decide by the typical or most-likely outcome of each strategy",
        "rule": {"kind": "token", "correct": "Fixed", "menu": ["Fixed", "Turbo"], "foils": ["Turbo"]},
    },
]

# The correct problem-type per probe (core/problem_types.py ids). The classify
# pass is graded against these. Mapping seeded from the golden-probe enum table.
CORRECT_TYPE = {
    "PC-01": "order_statistic_estimation",
    "PC-02": "conditional_probability_inversion",
    "PC-03": "aggregate_ratio_composition",
    "PC-04": "worst_case_bound",
    "PC-05": "constrained_schedule_construction",
    "PC-06": "event_driven_accumulation",
    "PC-07": "stratified_group_comparison",
    "PC-08": "eliminative_deduction",
    "PC-09": "aggregate_cost_minimization",
    "PC-10": "central_tendency_estimation",
}

# Cue-NEUTRAL twins (golden-probe draft §4-bis, post-verification: PC-09 "total"
# deleted, PC-10 "on average" -> "uses less time in total"). The classify pass
# runs on BOTH the cued `prompt` and this neutral twin; a classifier that scores
# high cued and low neutral is keyword-matching, not frame-selection.
NEUTRAL = {
    "PC-01": "A read request is sent to 10 replicas at once. The read is not done until the LAST of the 10 replies has come back. Each replica's reply arrives after a delay that is equally likely to be anywhere from 100 ms to 200 ms, independently of the others. If you ran this read over and over, what completion time would it come to on average, in milliseconds, rounded to the nearest millisecond?",
    "PC-02": "An anomaly alert fires on 99 of every 100 days that actually have an outage. On days with no outage it still fires on 2 of every 100. Real outages happen on 1 day in 500. The alert fired today. Across all the days on which the alert fires, on what fraction is there actually an outage? Give a percentage.",
    "PC-03": "A migration copies 8,000 MB. It moves the first 4,000 MB at 100 MB each second, then the last 4,000 MB at 25 MB each second. From the start of the copy to the end, how many MB did it move per second?",
    "PC-04": "Requests arrive and each one lands on one of 12 shards; you have no control over which shard any request lands on. You want to be sure that some shard has ended up with at least 3 requests on it. What is the smallest number of requests after which that is bound to be true, no matter how they happened to land?",
    "PC-05": "Five build tasks take 8, 7, 6, 5, and 4 hours. Three build machines run them in parallel; each task runs start-to-finish on one machine and cannot be paused or split. You may assign tasks to machines in whatever arrangement works best. How many hours pass until the last task finishes?",
    "PC-06": "A log volume starts empty and grows by 4 GB every hour. Every 6 hours a cleanup runs and removes 12 GB all at once; there is always at least 12 GB present when it runs. The volume holds 40 GB. How many hours pass before it is full for the first time?",
    "PC-07": "Two build pipelines ran last quarter. Atlas: frontend repos 81 of 87 builds passed; backend repos 192 of 263 passed. Borealis: frontend repos 234 of 270 builds passed; backend repos 55 of 80 passed. Your team builds only frontend repos and will send all of its builds to one pipeline. Which pipeline would give your team more passing builds? Choose Atlas or Borealis.",
    "PC-08": "A web stack has exactly four parts: a load balancer, an API service, a cache, and a database. Normal reads go load balancer -> API service -> cache -> database. Writes go load balancer -> API service -> database. Reads sent with the header X-Bypass-Cache go load balancer -> API service -> database. Since 14:00 today: all writes succeed; all normal reads time out; all reads sent with X-Bypass-Cache succeed; the cache's own health endpoint reports OK; the API service was redeployed at 13:35. Exactly one part is faulty. Which one? Choose load balancer, API service, cache, or database.",
    "PC-09": "A courier depot will be placed on a straight road. Five customer sites sit at mile markers 2, 5, 11, 19, and 23. Each day the courier drives from the depot out to each site and back, once per site. At which mile marker should the depot sit so that the distance driven each day comes out as small as possible?",
    "PC-10": "A nightly batch runs one of two ways. Fixed always takes exactly 60 minutes. Turbo takes 30 minutes on 7 nights out of 10, and 150 minutes on the other 3 nights out of 10. Over a long stretch of nights, which one uses less time in total? Choose Fixed or Turbo.",
}
