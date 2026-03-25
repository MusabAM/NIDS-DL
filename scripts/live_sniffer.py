import argparse
import time

import cicflowmeter.flow_session
import cicflowmeter.writer
import requests
from cicflowmeter.sniffer import create_sniffer
from cicflowmeter.writer import OutputWriter


def _format_key(k: str) -> str:
    """Heuristic to map cicflowmeter keys (e.g. flow_duration) to CICIDS2018 keys (e.g. Flow Duration)"""
    parts = k.split("_")
    return " ".join([p.capitalize() for p in parts])


class NIDSBackendWriter(OutputWriter):
    def __init__(self, output_url, model_type="CNN") -> None:
        self.url = output_url
        self.model_type = model_type
        self.session = requests.Session()
        self.flow_count = 0

    def write(self, data: dict) -> None:
        self.flow_count += 1

        # Transform data keys to match CICIDS2018 training column names heuristically
        mapped_features = {}
        for k, v in data.items():
            mapped_k = _format_key(k)
            # ensure value is numeric for inference
            try:
                mapped_features[mapped_k] = float(v)
            except:
                pass

        payload = {
            "features": mapped_features,
            "dataset_name": "CICIDS2018",
            "model_type": self.model_type,
        }

        try:
            resp = self.session.post(self.url, json=payload, timeout=2)
            if resp.status_code == 200:
                result = resp.json()
                print(
                    f"[Flow #{self.flow_count}] => Prediction: {result.get('prediction', 'Unknown')} (Confidence: {result.get('confidence', 0):.4f})"
                )
            else:
                print(f"[Flow #{self.flow_count}] => Error: {resp.status_code}")
        except Exception as e:
            print(f"[-] Backend connection failed: {e}")

    def __del__(self):
        self.session.close()


# Factory Override to inject into cicflowmeter
def custom_factory(output_mode, output):
    if output_mode == "nids":
        return NIDSBackendWriter(output)
    return cicflowmeter.writer.CSVWriter(output)


cicflowmeter.writer.output_writer_factory = custom_factory
cicflowmeter.flow_session.output_writer_factory = custom_factory


def main():
    parser = argparse.ArgumentParser(description="Live NIDS Packet Sniffer")
    parser.add_argument(
        "-i",
        "--interface",
        default=None,
        help="Network interface to sniff on (e.g. Wi-Fi). Omit to use default.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="CNN",
        help="Model to use for prediction (CNN, LSTM, Transformer, Autoencoder, Ensemble, Ensemble_Phase1)",
    )
    parser.add_argument(
        "-u",
        "--url",
        default="http://127.0.0.1:8000/api/predict/live",
        help="Backend prediction URL",
    )

    args = parser.parse_args()

    if not args.interface:
        try:
            from scapy.all import conf

            args.interface = conf.iface.name if conf.iface else conf.iface
        except ImportError:
            pass

    if args.interface:
        print(f"[*] Starting Sniffer on interface: {args.interface}")
    else:
        print("[*] Starting Sniffer on default interface (fallback)")

    print(
        f"[*] Extracting CICIDS2018 features locally and streaming to: {args.url} configured for {args.model}"
    )
    print("[*] Waiting for network flows to complete... (Produce some traffic!)")

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

        # Inject model type
        if hasattr(session, "output_writer") and isinstance(
            session.output_writer, NIDSBackendWriter
        ):
            session.output_writer.model_type = args.model

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
