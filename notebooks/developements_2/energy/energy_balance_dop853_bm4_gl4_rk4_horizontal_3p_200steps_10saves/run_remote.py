"""Execute the calculation notebook with live stream logs and persisted status."""
import json
import os
from pathlib import Path
import sys
import traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class LoggedNotebookClient(NotebookClient):
    """Mirror kernel streams into the detached process log."""

    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    directory = Path(__file__).resolve().parent
    root = next(p for p in directory.parents if (p/'pyproject.toml').is_file())
    os.chdir(root)
    status = directory/'execution_status.json'
    def write_status(state, **extra):
        temp = status.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state, pid=os.getpid(), **extra), indent=2))
        temp.replace(status)
    notebook = nbformat.read(directory/'calculation.ipynb', as_version=4)
    write_status('running')
    manager = KernelManager(kernel_name='python3')
    # Match the worker interpreter instead of an unrelated user kernel spec.
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    try:
        LoggedNotebookClient(notebook, km=manager, timeout=None,
                             resources={'metadata': {'path': str(root)}}).execute()
        if not (directory/'results.npz').is_file():
            raise RuntimeError('Notebook completed without results.npz.')
        write_status('complete', result='results.npz')
        print('STUDY COMPLETE: results.npz', flush=True)
    except BaseException as exc:
        write_status('failed', error=str(exc))
        traceback.print_exc()
        raise
    finally:
        nbformat.write(notebook, directory/'calculation_executed.ipynb')


if __name__ == '__main__':
    main()
