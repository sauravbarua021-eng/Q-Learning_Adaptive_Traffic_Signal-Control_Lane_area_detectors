import traci
import random
import numpy as np

# =========================
# SUMO CONFIG
# =========================
SUMO_CMD = ["sumo-gui", "-c", "osm.sumocfg"]

TLS_ID = "cluster_272554401_4123779839"

# Phases
PHASES = [0, 2]
YELLOW_PHASES = {0: 1, 2: 3}

PHASE_MEANING = {
    0: "Main Road Green",
    2: "Side Road Green"
}

# =========================
# DETECTORS
# =========================
det_A = ["e2_0", "e2_1", "e2_2"]
det_B = ["e2_3", "e2_4"]
det_C = ["e2_5", "e2_6", "e2_7"]

ALL_DETS = det_A + det_B + det_C

# =========================
# RL PARAMETERS
# =========================
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

Q = {}

# =========================
# STATE FUNCTION
# =========================
def get_state():
    def bucket(x):
        if x == 0:
            return 0
        elif x <= 5:
            return 1
        else:
            return 2

    a = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_A)
    b = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_B)
    c = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_C)

    return (bucket(a), bucket(b), bucket(c))


# =========================
# REWARD FUNCTION
# =========================
def get_reward():
    total_queue = sum(
        traci.lanearea.getLastStepHaltingNumber(d)
        for d in ALL_DETS
    )
    return -total_queue


# =========================
# ACTION SELECTION
# =========================
def choose_action(state):

    if state not in Q:
        Q[state] = {a: 0.0 for a in PHASES}

    # Exploration
    if random.random() < EPSILON:
        action = random.choice(PHASES)
        print(" >> Exploring")
        return action

    # Exploitation
    action = max(Q[state], key=Q[state].get)
    print(" >> Exploiting")
    return action


# =========================
# Q UPDATE
# =========================
def update_q(state, action, reward, next_state):

    if next_state not in Q:
        Q[next_state] = {a: 0.0 for a in PHASES}

    old_value = Q[state][action]
    next_max = max(Q[next_state].values())

    new_value = old_value + ALPHA * (
        reward + GAMMA * next_max - old_value
    )

    Q[state][action] = new_value

    return old_value, new_value


# =========================
# APPLY PHASE
# =========================
def apply_phase(action, green_time=10, yellow_time=3):

    # GREEN
    traci.trafficlight.setPhase(TLS_ID, action)
    for _ in range(green_time):
        traci.simulationStep()

    # YELLOW
    traci.trafficlight.setPhase(TLS_ID, YELLOW_PHASES[action])
    for _ in range(yellow_time):
        traci.simulationStep()


# =========================
# MAIN TRAINING LOOP
# =========================
def run_rl(episodes=5, steps_per_episode=3600):

    traci.start(SUMO_CMD)

    for ep in range(episodes):

        print("\n====================================")
        print(f"        EPISODE {ep+1} STARTED")
        print("====================================")

        step = 0

        while step < steps_per_episode:

            # -------- RAW DETECTOR VALUES --------
            a_raw = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_A)
            b_raw = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_B)
            c_raw = sum(traci.lanearea.getLastStepVehicleNumber(d) for d in det_C)

            # -------- CURRENT STATE --------
            state = get_state()

            print(f"\n[STEP {step}]")
            print(f" Detectors -> A:{a_raw}, B:{b_raw}, C:{c_raw}")
            print(f" State: {state}")

            # -------- ACTION --------
            action = choose_action(state)
            print(f" Action: {action} ({PHASE_MEANING[action]})")

            # -------- APPLY SIGNAL --------
            apply_phase(action)
            step += 13

            # -------- NEXT STATE + REWARD --------
            next_state = get_state()
            reward = get_reward()

            print(f" Next State: {next_state}")
            print(f" Reward (−queue): {reward}")

            # -------- Q UPDATE --------
            old_q, new_q = update_q(state, action, reward, next_state)

            print(f" Q-value: {old_q:.3f} → {new_q:.3f}")
            print(f" Max Q(state): {max(Q[state].values()):.3f}")

        print("\n====================================")
        print(f"       EPISODE {ep+1} COMPLETED")
        print("====================================")

    traci.close()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_rl()