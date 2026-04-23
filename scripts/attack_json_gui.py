import json
import random
import tkinter as tk
from tkinter import ttk, messagebox

# Full 76 features for CICIDS2018 (in correct training order)
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

NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

UNSW_NB15_COLUMNS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
    'sload', 'dload', 'sloss', 'dloss'
]

class AttackJsonGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NIDS-DL Attack JSON Generator (Novelty Edition)")
        self.root.geometry("650x700")
        self.root.configure(bg="#f3f4f6")
        
        style = ttk.Style()
        style.configure("TFrame", background="#f3f4f6")
        style.configure("TLabel", background="#f3f4f6", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1f2937")

        main_frame = ttk.Frame(root, padding="20 20 20 20")
        main_frame.pack(fill="both", expand=True)

        ttk.Label(main_frame, text="Network Attack Traffic Generator", style="Header.TLabel").pack(pady=(0, 20))

        # Dataset Selection
        ttk.Label(main_frame, text="1. Select Dataset:").pack(anchor="w", pady=(10, 5))
        self.dataset_var = tk.StringVar(value="CICIDS2018")
        self.dataset_cb = ttk.Combobox(main_frame, textvariable=self.dataset_var, state="readonly")
        self.dataset_cb['values'] = ["CICIDS2018", "NSL-KDD", "UNSW-NB15"]
        self.dataset_cb.pack(fill="x", pady=5)
        self.dataset_cb.bind("<<ComboboxSelected>>", self.update_attacks)

        # Attack Selection
        ttk.Label(main_frame, text="2. Select Attack Type:").pack(anchor="w", pady=(10, 5))
        self.attack_var = tk.StringVar()
        self.attack_cb = ttk.Combobox(main_frame, textvariable=self.attack_var, state="readonly")
        self.attack_cb.pack(fill="x", pady=5)
        self.attack_cb.bind("<<ComboboxSelected>>", self.generate_novel_json)

        # JSON Output
        txt_header = ttk.Frame(main_frame)
        txt_header.pack(fill="x", pady=(20, 5))
        ttk.Label(txt_header, text="3. Generated JSON (With High Novelty):").pack(side="left")
        self.feat_count_var = tk.StringVar(value="")
        ttk.Label(txt_header, textvariable=self.feat_count_var, font=("Segoe UI", 9, "italic"), foreground="#3b82f6").pack(side="right")

        self.json_text = tk.Text(main_frame, height=20, font=("Consolas", 10), bg="#ffffff", borderwidth=1, relief="solid", padx=10, pady=10)
        self.json_text.pack(fill="both", expand=True, pady=5)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(20, 0))

        ttk.Button(btn_frame, text="Generate New Variation", command=self.generate_novel_json).pack(side="left", padx=(0, 10))
        ttk.Button(btn_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side="left", padx=(0, 10))
        ttk.Button(btn_frame, text="Clear", command=self.clear_text).pack(side="left")

        self.update_attacks()

    def update_attacks(self, event=None):
        dataset = self.dataset_var.get()
        if dataset == "CICIDS2018":
            attacks = ["Benign", "DDoS Hulk", "Slowloris", "Brute Force SSH"]
        elif dataset == "NSL-KDD":
            attacks = ["Benign", "DoS (Smurf/Neptune)", "Probe"]
        else:
            attacks = ["Benign", "Generic Exploits", "Fuzzers"]
        
        self.attack_cb['values'] = attacks
        self.attack_var.set("")
        self.json_text.delete("1.0", tk.END)
        self.feat_count_var.set("")

    def generate_novel_json(self, event=None):
        dataset = self.dataset_var.get()
        attack = self.attack_var.get()
        
        if not dataset or not attack:
            return

        payload = {}
        
        if dataset == "CICIDS2018":
            payload = self._generate_cicids2018(attack)
            self.feat_count_var.set("76 Features Total")
        elif dataset == "NSL-KDD":
            payload = self._generate_nsl_kdd(attack)
            self.feat_count_var.set("41 Features Total")
        else:
            payload = self._generate_unsw(attack)
            self.feat_count_var.set("12 Features Total")

        json_str = json.dumps(payload, indent=2)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json_str)

    def _generate_cicids2018(self, attack):
        data = {col: 0.0 for col in CICIDS2018_COLUMNS}
        
        # Base templates with ranges for novelty
        if attack == "Benign":
            data['Flow Duration'] = random.randint(50000, 500000)
            data['Tot Fwd Pkts'] = random.randint(3, 15)
            data['Tot Bwd Pkts'] = random.randint(2, 12)
            data['Fwd Pkt Len Max'] = random.randint(40, 600)
            data['Bwd Pkt Len Max'] = random.randint(0, 1500)
            data['Flow Byts/s'] = round(random.uniform(500, 10000), 2)
            data['Flow Pkts/s'] = round(random.uniform(10, 200), 2)
            data['Init Fwd Win Byts'] = random.choice([255, 8192, 14600, 65535])
            data['Init Bwd Win Byts'] = random.choice([255, 8192, 14600, 65535])
            
        elif attack == "DDoS Hulk":
            data['Flow Duration'] = random.randint(30000000, 90000000)
            data['Tot Fwd Pkts'] = random.randint(2000, 20000)
            data['Tot Bwd Pkts'] = random.randint(2000, 20000)
            data['Fwd Pkt Len Max'] = 1460
            data['Bwd Pkt Len Max'] = 1460
            data['Flow Byts/s'] = round(random.uniform(1000000, 10000000), 2)
            data['Flow Pkts/s'] = round(random.uniform(500, 5000), 2)
            data['Fwd Header Len'] = data['Tot Fwd Pkts'] * 20
            data['Bwd Header Len'] = data['Tot Bwd Pkts'] * 20
            data['Init Fwd Win Byts'] = 8192
            data['Init Bwd Win Byts'] = 29200
            data['PSH Flag Cnt'] = 1
            data['ACK Flag Cnt'] = 1
            
        elif attack == "Slowloris":
            data['Flow Duration'] = random.randint(100000000, 120000000)
            data['Tot Fwd Pkts'] = random.randint(5, 12)
            data['Tot Bwd Pkts'] = random.randint(3, 8)
            data['Fwd Pkt Len Max'] = random.randint(20, 100)
            data['Flow Byts/s'] = round(random.uniform(0.1, 5.0), 4)
            data['Flow Pkts/s'] = round(random.uniform(0.01, 1.0), 4)
            data['Flow IAT Mean'] = round(data['Flow Duration'] / max(1, data['Tot Fwd Pkts']), 2)
            data['Idle Max'] = random.randint(5000000, 15000000)
            data['Init Fwd Win Byts'] = random.randint(1024, 8192)
            
        elif attack == "Brute Force SSH":
            data['Flow Duration'] = random.randint(1000000, 10000000)
            data['Tot Fwd Pkts'] = random.randint(20, 60)
            data['Tot Bwd Pkts'] = random.randint(20, 80)
            data['Fwd Pkt Len Mean'] = random.randint(30, 200)
            data['Bwd Pkt Len Mean'] = random.randint(50, 400)
            data['Flow Pkts/s'] = round(random.uniform(5.0, 50.0), 2)
            data['SYN Flag Cnt'] = 1
            data['PSH Flag Cnt'] = 1
            
        # Add a tiny bit of noise to all non-zero features for novelty
        for k, v in data.items():
            if v > 0 and isinstance(v, (int, float)):
                data[k] = round(v * random.uniform(0.95, 1.05), 4) if isinstance(v, float) else int(v * random.uniform(0.95, 1.05))
                
        return data

    def _generate_nsl_kdd(self, attack):
        data = {col: 0 for col in NSL_KDD_COLUMNS}
        data['protocol_type'] = 'tcp'
        data['service'] = 'http'
        data['flag'] = 'SF'
        
        if attack == "Benign":
            data['duration'] = 0
            data['src_bytes'] = random.randint(100, 1000)
            data['dst_bytes'] = random.randint(1000, 10000)
            data['logged_in'] = 1
            data['same_srv_rate'] = 1.0
            data['dst_host_srv_count'] = 255
            
        elif attack == "DoS (Smurf/Neptune)":
            data['service'] = random.choice(['icmp', 'private', 'http'])
            data['flag'] = random.choice(['S0', 'REJ'])
            data['count'] = random.randint(100, 511)
            data['serror_rate'] = round(random.uniform(0.8, 1.0), 2)
            data['dst_host_serror_rate'] = round(random.uniform(0.8, 1.0), 2)
            
        return data

    def _generate_unsw(self, attack):
        data = {col: 0.0 for col in UNSW_NB15_COLUMNS}
        if attack == "Benign":
            data['dur'] = round(random.uniform(0.01, 0.5), 3)
            data['rate'] = round(random.uniform(10.0, 500.0), 2)
            data['sttl'] = 31
            data['sload'] = round(random.uniform(1000.0, 100000.0), 1)
        else:
            data['dur'] = round(random.uniform(0.01, 1.5), 3)
            data['rate'] = round(random.uniform(0.1, 100.0), 2)
            data['sttl'] = random.choice([62, 254])
            data['sloss'] = random.randint(0, 5)
        return data

    def copy_to_clipboard(self):
        json_content = self.json_text.get("1.0", tk.END).strip()
        if not json_content: return
        self.root.clipboard_clear()
        self.root.clipboard_append(json_content)
        messagebox.showinfo("Success", "Novel JSON path copied to clipboard!")

    def clear_text(self):
        self.json_text.delete("1.0", tk.END)
        self.feat_count_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttackJsonGeneratorApp(root)
    root.mainloop()
