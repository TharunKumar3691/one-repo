import sys, time, nbformat
from nbclient import NotebookClient
nb = nbformat.read(sys.argv[1], as_version=4)
t = time.time()
client = NotebookClient(nb, timeout=3600, kernel_name="arcvenv", resources={"metadata": {"path": "/kaggle/working"}})
try:
    client.execute()
    ok = True
except Exception as e:
    ok = False; print("EXEC ERROR:", repr(e)[:2000])
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or "".join(o.get("traceback", []))
            print(txt[-3000:])
nbformat.write(nb, sys.argv[2])
print("OK" if ok else "FAILED", round(time.time() - t), "s")
