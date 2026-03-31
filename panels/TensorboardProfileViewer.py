# Comet Python Panel for visualizing Tensorboard Profile (and other) Data
# Log the tensorboard profile (and other data) with 
# >>> experiment.log_tensorflow_folder("./logs")

# NOTE: there is only one Tensorboard Server for your
# Python Panels; logs are shared across them

from comet_ml import API
import streamlit as st
import streamlit.components.v1 as components

import os
import json
import fcntl
import subprocess
import psutil
import time
import zipfile
import random
import signal

# --- Per-instance port assignment (6000-6009) ---
# Registry is stored in a file so it is shared across all panel processes
# and survives Streamlit session resets.

PORT_RANGE_START = 6000
PORT_RANGE_END = 6010  # exclusive
PORT_REGISTRY_FILE = "/tmp/tb_port_registry.json"


def get_instance_port(instance_id):
    """Return the port assigned to instance_id, assigning the next available
    port if this instance hasn't been seen before.  Uses a file lock so
    concurrent panel startups don't race.  Raises RuntimeError when the port
    range is exhausted."""
    if not os.path.exists(PORT_REGISTRY_FILE):
        with open(PORT_REGISTRY_FILE, "w") as f:
            json.dump({}, f)
    with open(PORT_REGISTRY_FILE, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            registry = json.load(f)
        except (json.JSONDecodeError, ValueError):
            registry = {}
        if instance_id not in registry:
            used_ports = set(registry.values())
            next_port = next(
                (p for p in range(PORT_RANGE_START, PORT_RANGE_END) if p not in used_ports),
                None,
            )
            if next_port is None:
                raise RuntimeError(
                    f"No available ports: all ports {PORT_RANGE_START}-{PORT_RANGE_END - 1} are in use."
                )
            registry[instance_id] = next_port
            f.seek(0)
            f.truncate()
            json.dump(registry, f)
        return registry[instance_id]


instance_id = os.environ.get("COMET_PANEL_INSTANCE_ID")
if instance_id is None:
    port = 6007
else:
    port = get_instance_port(instance_id)

st.set_page_config(layout="wide")

if "tensorboard_state" not in st.session_state:
    st.session_state["tensorboard_state"] = None

from streamlit_js_eval import get_page_location

api = API()
experiments = api.get_panel_experiments()

class EmptyExperiment:
    id = None
    name = ""


def select_experiment(experiment_list):
    names = [exp.name for exp in experiment_list]
    selected_idx = st.selectbox(
        "Select Experiment with log:",
        range(len(names)),
        format_func=lambda i: names[i],
    )
    return experiment_list[selected_idx]


experiments_with_log = [EmptyExperiment()]
for experiment in experiments:
    asset_list = experiment.get_asset_list("tensorflow-file")
    if asset_list:
        experiments_with_log.append(experiment)

if len(experiments_with_log) == 1:
    st.write("No experiments with log")
    st.stop()
elif len(experiments_with_log) == 2:
    selected_experiment = experiments_with_log[1]
else:
    selected_experiment = select_experiment(experiments_with_log)

if selected_experiment.id:
    page_location = get_page_location()
    if page_location is not None:
        if not os.path.exists("./%s" % selected_experiment.id):
            bar = st.progress(0, "Downloading log files...")
            selected_experiment.download_tensorflow_folder("./%s" % selected_experiment.id)
            bar.empty()
    
        selected_log = st.selectbox(
            "Select Profile to view:", 
            [""] + sorted(os.listdir("./%s/logs/" % selected_experiment.id))
        )
        if selected_log:
            command = f"/home/stuser/.local/bin/tensorboard --logdir ./{selected_experiment.id}/logs/{selected_log} --port {port}".split()
            env = {} # {"PYTHONPATH": "/.local/lib/python3.9/site-packages"}
            if st.session_state["tensorboard_state"] != (selected_experiment.id, selected_log):
                #print("Killing the hard way...")
                for process in psutil.process_iter():
                    try:
                        if "tensorboard" in process.exe():
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        print("Can't kill the server; continuing ...")
            
                process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
                st.session_state["tensorboard_state"] = (selected_experiment.id, selected_log)
                
                # Allow to start
                seconds = 5
                bar = st.progress(0, "Starting Tensorboard...")
                for i in range(seconds):
                    bar.progress(((i + 1) / seconds), "Starting Tensorboard...")
                    time.sleep(1)
                bar.empty()
    
            path, _ = page_location["pathname"].split("/component")
            url = page_location["origin"] + path + f"/port/{port}/server?x={random.randint(1,1_000_000)}#profile"
            st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
            components.iframe(src=url, height=700)
