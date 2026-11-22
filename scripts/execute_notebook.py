"""Execute a notebook with this Python environment and retain machine outputs."""
import argparse
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

parser = argparse.ArgumentParser()
parser.add_argument("notebook")
args = parser.parse_args()
path = Path(args.notebook).resolve()
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=2400,
                        resources={"metadata": {"path": str(path.parents[1])}})
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
print(f"Executed and saved {path.name}")
