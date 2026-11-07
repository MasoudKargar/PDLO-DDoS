from pathlib import Path
import random

from scapy.all import rdpcap, wrpcap


KEEP_PERCENTAGES = (5,10,20,40,55,90)


def create_versions(capture_path):
    capture = Path(capture_path)
    packets = rdpcap(str(capture))
    packet_count = len(packets)

    print(f"{capture.name}: {packet_count} packets")

    for percent in KEEP_PERCENTAGES:
        keep_count = int(packet_count * percent / 100)
        selected_indexes = sorted(random.sample(range(packet_count), keep_count))
        output_path = capture.with_name(
            f"{capture.stem}_{percent}percent{capture.suffix}"
        )

        selected_packets = [packets[index] for index in selected_indexes]
        wrpcap(str(output_path), selected_packets)
        print(f"  Created {output_path.name}: {keep_count} packets")


capture_paths = input(
    "Enter the paths of 2 pcap files, separated by commas: "
).split(",")
capture_paths = [path.strip() for path in capture_paths if path.strip()]

if len(capture_paths) != 2:
    raise ValueError("Please enter exactly 2 pcap file paths.")

for path in capture_paths:
    create_versions(path)
