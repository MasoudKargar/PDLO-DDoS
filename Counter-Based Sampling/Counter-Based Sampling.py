import os
import random
from scapy.all import rdpcap, wrpcap
from collections import Counter

def extract_key(pkt):
    """Extracts a simplified key: (src IP, dst IP, protocol)"""
    if pkt.haslayer("IP"):
        proto = pkt["IP"].proto
        src = pkt["IP"].src
        dst = pkt["IP"].dst
        return (src, dst, proto)
    return None

def reduce_pcap_balanced(input_path, output_path, keep_ratio, min_per_key=100):
    packets = rdpcap(input_path)
    keys = [extract_key(pkt) for pkt in packets]
    pattern_counter = Counter(keys)

    # Group packet indices by key
    key_to_indices = {}
    for i, key in enumerate(keys):
        if key is not None:
            key_to_indices.setdefault(key, []).append(i)

    selected_indices = set()

    # Step 1: Sample from each group
    for key, indices in key_to_indices.items():
        n = len(indices)
        keep_n = max(int(n * keep_ratio), min_per_key)
        keep_n = min(keep_n, n)
        selected = random.sample(indices, keep_n)
        selected_indices.update(selected)

    # Step 2: Fill up to target percentage if needed
    total_to_keep = int(len(packets) * keep_ratio)
    if len(selected_indices) < total_to_keep:
        additional_needed = total_to_keep - len(selected_indices)
        all_indices = set(range(len(packets)))
        remaining = list(all_indices - selected_indices)
        if additional_needed > 0 and remaining:
            selected_indices.update(random.sample(remaining, min(additional_needed, len(remaining))))

    # Final packet selection and save
    filtered_packets = [packets[i] for i in sorted(selected_indices)]
    wrpcap(output_path, filtered_packets)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pcap"):
            input_path = os.path.join(folder_path, filename)
            name, ext = os.path.splitext(filename)
            print(f"\nProcessing file: {filename}")

            for ratio in range(5, 100, 5):  # 5% to 95%
                keep_ratio = ratio / 100.0
                output_filename = f"{name}_{ratio}{ext}"
                output_path = os.path.join(folder_path, output_filename)
                print(f"  → Generating {output_filename} with {ratio}% of packets")
                reduce_pcap_balanced(input_path, output_path, keep_ratio)

if __name__ == "__main__":
    folder_path = "./XXX"  # Replace with your folder path
    process_folder(folder_path)
