import csv
import random

OUTPUT_FILE = "nsl_kdd_test_flows.csv"

HEADER = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "ftp_data", "ssh", "dns", "smtp", "telnet", "ecr_i"]
FLAGS = ["SF", "S0", "REJ", "RSTR"]


def normal_flow():
    return [
        random.randint(0, 10),
        "tcp",
        "http",
        "SF",
        random.randint(100, 2000),
        random.randint(1000, 10000),
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        random.randint(5, 50),
        random.randint(5, 50),
        0.0,
    ]


def dos_flow():
    return [
        0,
        random.choice(["tcp", "icmp"]),
        random.choice(["http", "ecr_i"]),
        "S0",
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        random.randint(300, 511),
        random.randint(300, 511),
        round(random.uniform(0.8, 1.0), 2),
    ]


def probe_flow():
    return [
        random.randint(0, 2),
        "tcp",
        random.choice(SERVICES),
        random.choice(["S0", "REJ"]),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        random.randint(100, 250),
        random.randint(10, 50),
        round(random.uniform(0.6, 0.9), 2),
    ]


def r2l_flow():
    return [
        random.randint(2, 20),
        "tcp",
        random.choice(["ftp", "smtp", "telnet"]),
        "SF",
        random.randint(200, 1500),
        random.randint(1000, 6000),
        0,
        0,
        0,
        random.randint(1, 5),
        random.randint(1, 3),
        0,
        random.randint(1, 3),
        0,
        1,
        1,
        random.randint(1, 3),
        1,
        2,
        0,
        0,
        1,
        random.randint(10, 30),
        random.randint(2, 10),
        round(random.uniform(0.1, 0.4), 2),
    ]


def u2r_flow():
    return [
        random.randint(5, 50),
        "tcp",
        "telnet",
        "SF",
        random.randint(500, 3000),
        random.randint(5000, 15000),
        0,
        0,
        0,
        random.randint(3, 10),
        1,
        1,
        random.randint(3, 10),
        1,
        1,
        random.randint(3, 10),
        random.randint(2, 5),
        random.randint(1, 3),
        random.randint(2, 5),
        0,
        0,
        0,
        random.randint(5, 15),
        random.randint(2, 5),
        round(random.uniform(0.05, 0.2), 2),
    ]


def generate_flows(num_records=100):
    flows = []
    for _ in range(num_records):
        traffic_type = random.choice(
            [normal_flow, dos_flow, probe_flow, r2l_flow, u2r_flow]
        )
        flows.append(traffic_type())
    return flows


def main():
    flows = generate_flows(200)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(flows)

    print(f"[+] Generated {len(flows)} NSL-KDD test flows")
    print(f"[+] Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
