import json
import random
import argparse
import sys

# Full feature lists
CICIDS2018_COLUMNS = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
    'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

def generate_cicids2018(attack):
    data = {col: 0.0 for col in CICIDS2018_COLUMNS}
    attack = attack.lower()
    
    if attack == "benign":
        data['Flow Duration'] = random.randint(50000, 500000)
        data['Tot Fwd Pkts'] = random.randint(3, 15)
        data['Tot Bwd Pkts'] = random.randint(2, 12)
        data['Flow Byts/s'] = round(random.uniform(500, 10000), 2)
    elif attack == "ddos":
        data['Flow Duration'] = random.randint(30000000, 90000000)
        data['Tot Fwd Pkts'] = random.randint(5000, 15000)
        data['Tot Bwd Pkts'] = random.randint(5000, 15000)
        data['Flow Byts/s'] = round(random.uniform(1000000, 5000000), 2)
        data['Init Fwd Win Byts'] = 8192
        data['Init Bwd Win Byts'] = 29200
    elif attack == "slowloris":
        data['Flow Duration'] = random.randint(100000000, 120000000)
        data['Flow Byts/s'] = round(random.uniform(0.1, 5.0), 4)
        data['Idle Max'] = random.randint(5000000, 15000000)
    elif attack == "bruteforce":
        data['Flow Duration'] = random.randint(1000000, 10000000)
        data['Tot Fwd Pkts'] = random.randint(20, 60)
        data['Flow Pkts/s'] = round(random.uniform(5.0, 50.0), 2)
    
    # Noise
    for k, v in data.items():
        if v > 0 and isinstance(v, (int, float)):
            data[k] = round(v * random.uniform(0.95, 1.05), 4) if isinstance(v, float) else int(v * random.uniform(0.95, 1.05))
            
    return data

def main():
    parser = argparse.ArgumentParser(description='Generate Novel Attack JSON for NIDS-DL')
    parser.add_argument('--dataset', type=str, choices=['CICIDS2018', 'NSL-KDD', 'UNSW-NB15'], default='CICIDS2018')
    parser.add_argument('--attack', type=str, required=True)
    args = parser.parse_args()
    
    if args.dataset == 'CICIDS2018':
        payload = generate_cicids2018(args.attack)
    else:
        # Fallback for brevity in script
        payload = {"info": "Use the GUI for full support of this dataset's novelty patterns."}
        
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
