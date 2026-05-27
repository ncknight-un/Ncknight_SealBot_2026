# source venv/bin/activate

import pandas as pd
import json
import matplotlib.pyplot as plt

# ==== Load CSV ====
df = pd.read_csv("PoolTest3_Bag3.csv")

# ==== Filter only VFR_HUD messages ====
vfr = df[df["msg_type"] == "VFR_HUD"].copy()

# ==== Parse JSON in 'data' column ====
def parse_json(cell):
    try:
        return json.loads(cell)
    except:
        return {}

vfr_parsed = vfr["data"].apply(parse_json)

# ==== Extract fields ====
vfr["timestamp"] = vfr["system_time"]
vfr["airspeed"] = vfr_parsed.apply(lambda x: x.get("airspeed"))
vfr["alt"] = vfr_parsed.apply(lambda x: x.get("alt"))
vfr["climb"] = vfr_parsed.apply(lambda x: x.get("climb"))
vfr["groundspeed"] = vfr_parsed.apply(lambda x: x.get("groundspeed"))
vfr["heading"] = vfr_parsed.apply(lambda x: x.get("heading"))
vfr["throttle"] = vfr_parsed.apply(lambda x: x.get("throttle"))

# ==== Plot each param ====
params = ["airspeed", "alt", "climb", "groundspeed", "heading", "throttle"]

# for p in params:
#     plt.figure()
#     plt.plot(vfr["timestamp"], vfr[p])
#     plt.xlabel("Timestamp")
#     plt.ylabel(p)
#     plt.title(f"VFR_HUD - {p} vs Time")
    # plt.show()

# ==== Plot all params together as subplots ====
fig, axs = plt.subplots(len(params), 1, figsize=(10, 15), sharex=True)

for i, p in enumerate(params):
    axs[i].plot(vfr["timestamp"], vfr[p])
    axs[i].set_ylabel(p)
    axs[i].grid(True)

axs[-1].set_xlabel("Timestamp")
plt.suptitle("VFR_HUD Parameters vs Time", y=0.92)
plt.tight_layout()
plt.show()