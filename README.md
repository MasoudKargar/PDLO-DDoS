# PDLO-DDoS: Probabilistic Data Lifespan Optimization for Edge-Assisted IoT DDoS Detection

Code for the paper **"A lightweight convolutional framework using probabilistic data lifespan optimization for edge-assisted IoT DDoS detection"** (submitted to *Scientific Reports*).

PDLO reduces the volume of network traffic before training a small CNN. Packets are grouped by flow (source IP, destination IP, protocol) and a fraction gamma of each flow is kept, so that high-volume floods are pruned while low-frequency flows stay represented. The name "lifespan" refers to how much of each flow is retained. The method uses packet counts only, not timestamps or packet age, so it is a frequency-based flow sampling scheme.

## Pipeline

1. **Stage 1, counter-based flow sampling** (`counter_based_sampling/counter_based_sampling.py`)
2. **Stage 2, preprocessing** (`cnn_ddos_detection/lucid_dataset_parser.py`): 10 s windows, 11 per-packet features, 10 x 11 matrices
3. **Stage 3, flow-aware 80/10/10 split** (same parser, `--preprocess_folder` step)
4. **Stage 4/5, training and test** (`cnn_ddos_detection/lucid_cnn.py`): grid search, early stopping, test metrics, FLOPs, latency

## Repository structure

```
├── counter_based_sampling/
│   └── counter_based_sampling.py    # Stage 1
├── cnn_ddos_detection/
│   ├── lucid_cnn.py                 # model, grid search, training, testing
│   ├── lucid_dataset_parser.py      # Stages 2 and 3
│   ├── util_functions.py
│   ├── README.md                    # original LUCID documentation
│   ├── sample-dataset/
│   └── output/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Stage 1: counter-based flow sampling

For each pcap file, packets are grouped by the key (source IP, destination IP, protocol). With N(k) packets in flow k, the number kept is

```
R(k) = min( N(k), max( floor(N(k) * gamma), m ) ),   m = 100
```

- gamma is in [0.05, 0.95]. Flows with at most 100 packets are kept in full at every gamma.
- If the per-flow pass keeps fewer than floor(gamma * total packets), extra packets are drawn uniformly at random from the unselected ones until that total is reached. So gamma is the minimum fraction retained, not the exact fraction.
- Sampling within a flow is uniform without replacement (`random.sample`).
- Each pcap file is sampled independently. Packets without an IP header are not assigned to a flow and can only be kept by the fill-up step.
- **The random generator is not seeded, so Stage 1 output is not exactly reproducible.**

### Usage

```bash
cd counter_based_sampling
# edit folder_path in counter_based_sampling.py, then:
python counter_based_sampling.py
```

Or from Python:

```python
from counter_based_sampling import reduce_pcap_balanced, process_folder

reduce_pcap_balanced("input.pcap", "output.pcap", keep_ratio=0.1, min_per_key=100)
process_folder("./pcap_files/")
```

## Stages 2 and 3: preprocessing and split

Settings used in the paper for all three datasets:

- Sub-flows are keyed by the bidirectional 5-tuple (source IP, source port, destination IP, destination port, protocol).
- Time window t = 10 s. Windows are global within a pcap file. There is no early termination on TCP FIN/RST and no idle timeout.
- Sequence length n = 10 packets. Longer sub-flows are truncated, shorter ones are zero-padded (padding is not masked).
- 11 per-packet header features: relative timestamp, packet length, highest protocol layer, IP flags, protocol bitmask, TCP length, TCP acknowledgment number, TCP flags, TCP window size, UDP length, ICMP type. Features are Min-Max scaled with fixed nominal bounds, so no statistic comes from the validation or test data.
- Labels come from a fixed list of attacker and victim IPs per dataset.
- Benign and DDoS flows are balanced towards 1:1, then whole flows are assigned to train, validation and test in an 80:10:10 ratio. The split is not class-stratified. Fixed seed 1.

```bash
cd cnn_ddos_detection
python lucid_dataset_parser.py --dataset_type CUSTOM --dataset_folder ./your-data/ \
    --packets_per_flow 10 --dataset_id YOUR_DATASET --traffic_type all --time_window 10
python lucid_dataset_parser.py --preprocess_folder ./your-data/
```

## Stage 4 and 5: model and training

The model (LDCNN) is a single convolutional layer followed by global max pooling and one sigmoid unit:

- Input: 10 x 11 matrix, reshaped to 10 x 11 x 1
- `Conv2D` with k filters and a 3 x 11 kernel (equivalent to a 1D convolution of length 3 over 11 channels), L1 or L2 kernel regularizer (strength 0.01)
- Dropout (applied before ReLU), then ReLU
- Global max pooling
- `Dense(1, sigmoid)`, threshold 0.5

Hyperparameter grid (252 configurations), same for all datasets and retention levels:

| Parameter | Values |
| --- | --- |
| Filters k | 1, 2, 4, 8, 16, 32, 64 |
| Dropout | 0.2, 0.3, 0.4 |
| Batch size | 1024, 2048 |
| Regularization | L1, L2 |
| Learning rate | 0.001, 0.01, 0.1 |

Each configuration is scored by 2-fold cross-validation accuracy on the training set (`GridSearchCV`). The best one is refitted on the training set with Adam, binary cross-entropy, at most 50 epochs, and early stopping on validation loss (patience 10, best weights restored). Python, NumPy and TensorFlow are seeded with 1.

```bash
python lucid_cnn.py   # add the exact arguments used for training and testing
```

## Full-data baseline (LUCID)

LUCID (Doriguzzi-Corin et al., IEEE TNSM 2020) uses the same layer structure as the LDCNN. In the paper it is trained and tested on 100% of each dataset on the same machine, so it serves as the same CNN without Stage 1 sampling. Preprocessing, batch size and hyperparameters may differ from the PDLO runs.

## Installation

Tested with Python 3.9+ and TensorFlow 2.12. A GPU is optional.

```bash
git clone https://github.com/MasoudKargar/PDLO-DDoS.git
cd PDLO-DDoS
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pcap parsing needs `tshark`: `sudo apt-get install tshark` (Linux), `brew install wireshark` (macOS), or Wireshark for Windows.

## Datasets

| Dataset | Source |
| --- | --- |
| CIC-DDoS2019 | https://www.unb.ca/cic/datasets/ddos-2019.html |
| Edge-IIoT Cyber Security | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot |
| CoAP-DDoS | Kaggle Repository |

The datasets are not redistributed here.

## Reported metrics

Accuracy, F1, TPR, FPR, TNR, FNR, FLOPs, and inference time per sample. Timings in the paper were measured on a laptop (AMD Ryzen 7 5800H, 16 GB RAM, NVIDIA RTX 3050 Ti), not on physical edge hardware.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper and the Zenodo archive.

## License

Counter-based sampling and PDLO code: ADD LICENSE (for example MIT). The CNN code is derived from LUCID and keeps its Apache License 2.0.

## Acknowledgments

Based on LUCID: R. Doriguzzi-Corin, S. Millar, S. Scott-Hayward, J. Martinez-del-Rincon, D. Siracusa, "Lucid: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," IEEE TNSM, vol. 17, pp. 876-889, 2020.
