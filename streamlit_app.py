
import streamlit as st
import nbformat
import io, sys, contextlib, types, traceback, os
from pathlib import Path

st.set_page_config(page_title="Notebook Stepper UI", layout="wide")
st.title("📓 Notebook Stepper UI")
st.caption("Run notebook code cells in a guided, step-by-step UI.")

# --- Sidebar: Notebook selection ---
default_nb = st.sidebar.text_input(
    "Notebook path (.ipynb)",
    value=str(Path(r"D:\Install\HQ_Transfer\Manuals\Programming examples\Commandserver\Python\final_workflow.ipynb")),
    help="Change this to your .ipynb if needed."
)
run_fresh = st.sidebar.button("Reload notebook")

# --- Load notebook ---
@st.cache_data(show_spinner=False)
def load_nb(path: str):
    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    titles = []
    for c in code_cells:
        # title = first non-empty line, prefer comment line
        title = None
        for line in c.source.splitlines():
            t = line.strip()
            if not t:
                continue
            if t.startswith("#"):
                title = t.lstrip("#").strip()
            else:
                title = t
            break
        if not title:
            title = "(empty code cell)"
        titles.append(title[:120])
    return code_cells, titles

if not os.path.exists(default_nb):
    st.error(f"Notebook not found: {default_nb}")
    st.stop()

if run_fresh:
    load_nb.clear()

code_cells, titles = load_nb(default_nb)

# --- Session state: execution namespace & progress ---
if "ns" not in st.session_state:
    st.session_state.ns = {}  # shared exec namespace across cells
if "executed" not in st.session_state:
    st.session_state.executed = [False] * len(code_cells)
if "current" not in st.session_state:
    st.session_state.current = 0

# Utilities
def reset_state():
    st.session_state.ns = {}
    st.session_state.executed = [False] * len(code_cells)
    st.session_state.current = 0

def mark_executed(idx: int):
    st.session_state.executed[idx] = True

def can_advance_to(idx: int) -> bool:
    # Only allow going to idx if all previous cells executed
    return all(st.session_state.executed[:idx])

st.sidebar.markdown("### Controls")
if st.sidebar.button("🔁 Reset session"):
    reset_state()
    st.experimental_rerun()

# Visual progress
done_count = sum(st.session_state.executed)
st.progress(done_count / max(1, len(code_cells)))
st.write(f"**Cells executed:** {done_count} / {len(code_cells)}")

# --- Cell navigation ---
# Enforce sequential order by disabling selection of cells beyond first unexecuted.
first_unrun = next((i for i, x in enumerate(st.session_state.executed) if not x), len(code_cells)-1)
st.session_state.current = first_unrun

# Show table of contents
with st.expander("Step list (read-only order)", expanded=True):
    for i, t in enumerate(titles):
        prefix = "✅" if st.session_state.executed[i] else f"{i}️⃣"
        st.write(f"{prefix} **Step {i}:** {t}")

idx = st.session_state.current
st.subheader(f"Step {idx}: {titles[idx]}")

cell = code_cells[idx]
editable = st.toggle("Edit this cell before running", value=False, help="Optionally tweak the code.")
src = cell.source
if editable:
    src = st.text_area("Code", value=src, height=260)
else:
    st.code(src, language="python")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    run_btn = st.button("▶️ Run this step", type="primary")
with col2:
    run_to_btn = st.button("⏭ Run all up to this step")
with col3:
    skip_btn = st.button("✅ Mark as done (no execution)")

# Shared capture
class Tee(io.StringIO):
    def __init__(self, *streams):
        super().__init__()
        self.streams = streams
    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return super().write(s)

def exec_cell(code: str):
    # Execute cell in shared namespace; capture stdout/stderr
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, out_buf)
    sys.stderr = Tee(sys.stderr, err_buf)
    try:
        exec(compile(code, f"<cell {idx}>", "exec"), st.session_state.ns, st.session_state.ns)
        return out_buf.getvalue(), err_buf.getvalue(), None
    except Exception as e:
        tb = traceback.format_exc()
        return out_buf.getvalue(), err_buf.getvalue(), tb
    finally:
        sys.stdout, sys.stderr = old_out, old_err

ran_any = False

if run_to_btn:
    # Run all from first unexecuted up to idx, sequentially
    start = next((i for i, x in enumerate(st.session_state.executed) if not x), idx)
    for j in range(start, idx+1):
        cout, cerr, err = exec_cell(code_cells[j].source)
        st.markdown(f"**Output for Step {j}:**")
        if cout.strip():
            st.code(cout)
        if cerr.strip():
            st.warning("stderr:")
            st.code(cerr)
        if err:
            st.error("Exception while executing:")
            st.code(err)
            st.stop()
        st.session_state.executed[j] = True
        ran_any = True

if run_btn:
    cout, cerr, err = exec_cell(src)
    st.markdown("**Output:**")
    if cout.strip():
        st.code(cout)
    if cerr.strip():
        st.warning("stderr:")
        st.code(cerr)
    if err:
        st.error("Exception while executing:")
        st.code(err)
    else:
        mark_executed(idx)
        ran_any = True

if skip_btn:
    mark_executed(idx)
    ran_any = True

# Advance automatically if this step is done
if ran_any:
    # Move to next unexecuted step if exists
    if done_count + 1 < len(code_cells):
        st.experimental_rerun()
    else:
        st.success("🎉 All steps completed!")

# --- Advanced: inspect / export namespace ---
with st.expander("🧠 Execution namespace (debug)"):
    keys = sorted(k for k in st.session_state.ns.keys() if not k.startswith("__"))
    st.write(f"Variables: {len(keys)}")
    for k in keys:
        try:
            v = st.session_state.ns[k]
            st.write(f"- **{k}** = `{repr(v)[:120]}`")
        except Exception as e:
            st.write(f"- **{k}** = <unreprable: {e}>")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: run with `streamlit run streamlit_app.py`")
