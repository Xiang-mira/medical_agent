from __future__ import annotations

from scheduler.resource_discovery import parse_scontrol_nodes, parse_sinfo_fallback


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


def test_parse_sinfo_fallback():
    rows = parse_sinfo_fallback("cpu|up|10-00:00:00|143\ngpu|up|10-00:00:00|19\n")
    assert rows["cpu"]["availability"] == "up"
    assert rows["gpu"]["nodes"] == "19"
