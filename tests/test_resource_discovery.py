from __future__ import annotations

from scheduler.resource_discovery import parse_scontrol_job, parse_scontrol_nodes, parse_sinfo_fallback, validate_snapshot_invariants


def test_parse_scontrol_nodes_computes_gpu_shapes():
    text = """
NodeName=t4-01 Arch=x86 CoresPerSocket=1
   State=MIXED CPUTot=64 CPUAlloc=8 RealMemory=250000 AllocMem=1000 Gres=gpu:T4:6 CfgTRES=cpu=64,mem=250000M,gres/gpu=6 AllocTRES=cpu=8,mem=1000M,gres/gpu=2 Partitions=gpu
NodeName=a100-01 Arch=x86 CoresPerSocket=1
   State=IDLE CPUTot=64 CPUAlloc=0 RealMemory=500000 AllocMem=0 Gres=gpu:A100:2 CfgTRES=cpu=64,mem=500000M,gres/gpu=2 AllocTRES= Partitions=gpua100
NodeName=h100-01 Arch=x86 CoresPerSocket=1
   State=DRAIN CPUTot=64 CPUAlloc=0 RealMemory=500000 AllocMem=0 Gres=gpu:H100:4 CfgTRES=cpu=64,mem=500000M,gres/gpu=4 AllocTRES= Partitions=gpuh100
"""
    nodes = parse_scontrol_nodes(text)
    assert nodes[0]["gpus_free"] == 4
    assert nodes[1]["gpu_type"] == "A100"
    assert nodes[1]["gpus_free"] == 2
    assert "DRAIN" in nodes[2]["state"]
    assert nodes[2]["schedulable"] is False


def test_parse_sinfo_fallback():
    rows = parse_sinfo_fallback("cpu|up|10-00:00:00|143\ngpu|up|10-00:00:00|19\n")
    assert rows["cpu"]["availability"] == "up"
    assert rows["gpu"]["nodes"] == "19"


def test_parse_pending_job_gpu_demand_from_req_tres_and_tres_per_node():
    job = parse_scontrol_job("JobId=4012308 JobState=PENDING Reason=QOSMaxGRESPerUser Partition=gpuh100 NumNodes=1 ReqTRES=cpu=127,mem=850G,node=1,billing=127,gres/gpu=4,gres/gpu:h100=4 TresPerNode=gres/gpu:H100:4")
    assert job["gpu_per_job"] == 4
    assert job["shape"] == "4 GPUs on one node"
    assert job["reason"] == "QOSMaxGRESPerUser"


def test_snapshot_invariants():
    snap = {"partitions": {"gpu": {"gpu_type": "T4", "physical_configured_total": 114, "allocatable_configured_total": 108, "allocated_estimate": 10, "idle_estimate": 98}}}
    assert validate_snapshot_invariants(snap)["status"] == "success"
    bad = {"partitions": {"gpu": {"gpu_type": "T4", "physical_configured_total": 1, "allocatable_configured_total": 1, "allocated_estimate": 2, "idle_estimate": 0}}}
    assert validate_snapshot_invariants(bad)["status"] == "failed"
