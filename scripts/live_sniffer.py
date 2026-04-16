import argparse
import time

import cicflowmeter.flow_session
import cicflowmeter.writer
import requests
from cicflowmeter.sniffer import create_sniffer
from cicflowmeter.writer import OutputWriter


# ---------------------------------------------------------------------------
# Key formatters — cicflowmeter outputs underscore_lower keys by default
# These heuristics map them to the feature names each dataset expects
# ---------------------------------------------------------------------------

def _format_key_cicids(k: str) -> str:
    """Maps e.g. 'flow_duration' -> 'Flow Duration' (CICIDS2017 / CICIDS2018 style)."""
    parts = k.split("_")
    return " ".join([p.capitalize() for p in parts])


def _format_key_unsw(k: str) -> str:
    """Maps e.g. 'flow_duration' -> 'dur', 'total_fwd_pkts' -> 'spkts' etc.
    cicflowmeter does not produce UNSW features natively; we pass raw keys and
    let the backend scaler/preprocessor drop unknowns and fill with defaults."""
    return k.lower()


def _format_key_nslkdd(k: str) -> str:
    """NSL-KDD uses snake_case feature names."""
    return k.lower()


DATASET_KEY_FORMATTERS = {
    "CICIDS2018": _format_key_cicids,
    "CICIDS2017": _format_key_cicids,
    "UNSW-NB15":  _format_key_unsw,
    "NSL-KDD":    _format_key_nslkdd,
}


class NIDSBackendWriter(OutputWriter):
    def __init__(self, output_url: str, model_type: str = "CNN", dataset: str = "CICIDS2018") -> None:
        self.url = output_url
        self.model_type = model_type
        self.dataset = dataset
        self.session = requests.Session()
        self.flow_count = 0
        self._key_fmt = DATASET_KEY_FORMATTERS.get(dataset, _format_key_cicids)

    def write(self, data: dict) -> None:
        self.flow_count += 1

        # Transform raw cicflowmeter keys to the target dataset's naming convention
        mapped_features = {}
        for k, v in data.items():
            mapped_k = self._key_fmt(k)
            try:
                mapped_features[mapped_k] = float(v)
            except (ValueError, TypeError):
                pass  # drop non-numeric values; backend preprocessor handles missing cols

        payload = {
            "features": mapped_features,
            "dataset_name": self.dataset,
            "model_type": self.model_type,
        }

        try:
            resp = self.session.post(self.url, json=payload, timeout=2)
            if resp.status_code == 200:
                result = resp.json()
                pred = result.get("prediction", "Unknown")
                conf = result.get("confidence", 0)
                zero_day = result.get("zeroDayPossible", False)
                tag = " [POSSIBLE ZERO-DAY]" if zero_day else ""
                print(
                    f"[Flow #{self.flow_count}] [{self.dataset}|{self.model_type}] "
                    f"=> {pred} (conf: {conf:.4f}){tag}"
                )
            else:
                print(f"[Flow #{self.flow_count}] => HTTP Error: {resp.status_code}")
        except Exception as e:
            print(f"[-] Backend connection failed: {e}")

    def __del__(self):
        self.session.close()


# ---------------------------------------------------------------------------
# cicflowmeter factory override
# ---------------------------------------------------------------------------

def custom_factory(output_mode, output):
    if output_mode == "nids":
        return NIDSBackendWriter(output)
    return cicflowmeter.writer.CSVWriter(output)


cicflowmeter.writer.output_writer_factory = custom_factory
cicflowmeter.flow_session.output_writer_factory = custom_factory


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live NIDS Packet Sniffer")
    parser.add_argument(
        "-i", "--interface",
        default=None,
        help="Network interface to sniff on (e.g. Wi-Fi). Omit to use system default.",
    )
    parser.add_argument(
        "-m", "--model",
        default="CNN",
        help=(
            "Model type for prediction. Options: CNN, LSTM, Transformer, Autoencoder, VQC, "
            "Ensemble (all models), Ensemble_Phase1 (CNN+LSTM+Transformer only), "
            "Ensemble_AE (Phase1 + Autoencoder), Ensemble_VQC (Phase1 + VQC)"
        ),
    )
    parser.add_argument(
        "-d", "--dataset",
        default="CICIDS2018",
        choices=["CICIDS2018", "CICIDS2017", "NSL-KDD", "UNSW-NB15"],
        help="Dataset/feature schema to use when submitting flows to the backend.",
    )
    parser.add_argument(
        "-u", "--url",
        default="http://127.0.0.1:8000/api/predict/live",
        help="Backend prediction endpoint URL.",
    )

    args = parser.parse_args()

    # Resolve interface
    if not args.interface:
        try:
            from scapy.all import conf
            args.interface = conf.iface.name if conf.iface else conf.iface
        except ImportError:
            pass

    iface_label = args.interface if args.interface else "system default"
    print(f"[*] Starting Sniffer on interface: {iface_label}")
    print(f"[*] Dataset: {args.dataset}  |  Model: {args.model}")
    print(f"[*] Streaming flows -> {args.url}")
    print("[*] Waiting for network flows to complete... (generate some traffic!)")

    try:
        sniffer, session = create_sniffer(
            input_file=None,
            input_interface=args.interface,
            output_mode="nids",
            output=args.url,
            input_directory=None,
            fields=None,
            verbose=False,
        )

        # Inject dataset + model into the writer instance
        if hasattr(session, "output_writer") and isinstance(session.output_writer, NIDSBackendWriter):
            session.output_writer.model_type = args.model
            session.output_writer.dataset = args.dataset
            session.output_writer._key_fmt = DATASET_KEY_FORMATTERS.get(args.dataset, _format_key_cicids)

        sniffer.start()
        sniffer.join()

    except KeyboardInterrupt:
        print("\n[*] Stopping sniffer...")
        if "sniffer" in locals():
            sniffer.stop()
            sniffer.join()
    except Exception as e:
        print(f"\n[-] Fatal Sniffer Error: {e}")
    finally:
        if "session" in locals():
            session.flush_flows()
        print("[*] Done.")


if __name__ == "__main__":
    main()
