# Comet Python Panel for visualizing Tensorboard Data by Group
# >>> experiment.log_other("Group", "GROUP-NAME")
# >>> experiment.log_tensorflow_folder("./logs")
# In the UI, group on "Group"

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
import glob
import shutil

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

log_dir = f"./logs/{instance_id or 'default'}"
cache_dir = f"./tb_cache/{instance_id or 'default'}"

st.set_page_config(layout="wide")

from streamlit_js_eval import get_page_location

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

DEBUG = False

# Clear cache and downloads
if DEBUG:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

api = API()
experiments = api.get_panel_experiments()

needs_refresh = False
page_location = get_page_location()
if page_location is not None:
    if True:
        column = st.columns([.7, .3])
        clear = column[1].checkbox("Clear previous logs", value=True)
        if column[0].button("Copy Selected Experiment Logs to Tensorboard Server", type="primary"):
            needs_refresh = True
            if clear and os.path.exists(log_dir):
                for filename in glob.glob(f"{log_dir}/*"):
                    shutil.move(filename, cache_dir)
            bar = st.progress(0, "Downloading log files...")
            for i, experiment in enumerate(experiments):
                bar.progress(i/len(experiments), "Downloading log files...")
                if not os.path.exists(f"{log_dir}/{experiment.name}"):
                    if os.path.exists(f"{cache_dir}/{experiment.name}"):
                        if DEBUG: print("found in cache!")
                        shutil.move(
                            f"{cache_dir}/{experiment.name}",
                            f"{log_dir}/{experiment.name}",
                        )
                    else:
                        if DEBUG: print("downloading...")
                        assets = experiment.get_asset_list("tensorflow-file")
                        if assets:
                            if DEBUG: print(assets[0]["fileName"])
                            if assets[0]["fileName"].startswith("logs/"):
                                experiment.download_tensorflow_folder("./")
                                downloaded = f"./logs/{experiment.name}"
                                if os.path.exists(downloaded):
                                    shutil.move(downloaded, f"{log_dir}/{experiment.name}")
                            else:
                                experiment.download_tensorflow_folder(f"{log_dir}/")
            bar.empty()

        running = False
        for process in psutil.process_iter():
            try:
                if "tensorboard" in process.exe():
                    running = True
            except:
                pass
        if not running:
            command = f"/home/stuser/.local/bin/tensorboard --logdir {log_dir} --port {port}".split()
            env = {} # {"PYTHONPATH": "/home/st_user/.local/lib/python3.9/site-packages"}
            process = subprocess.Popen(command, preexec_fn=os.setsid, env=env)
            needs_refresh = True

        if needs_refresh:
            # Allow to start/update
            seconds = 5
            bar = st.progress(0, "Updating Tensorboard...")
            for i in range(seconds):
                bar.progress(((i + 1) / seconds), "Updating Tensorboard...")
                time.sleep(1)
            bar.empty()

        path, _ = page_location["pathname"].split("/component")
        url = page_location["origin"] + path + f"/port/{port}/server?x={random.random()}"
        st.markdown('<a href="%s" style="text-decoration: auto;">⛶ Open in tab</a>' % url, unsafe_allow_html=True)
        components.iframe(src=url, height=700)
