"""


ONE rigid formation structure with Pure Pursuit navigation:
- FormationController navigates CENTER along waypoints using Pure Pursuit
- Each Vehicle reads center + applies fixed offset rotated by heading
- Formation shape stays rigid through turns (triangle stays triangle)
- Heading smoothly follows path tangent — realistic AUV rotation

Ports: 8000 stream, 8001 commands
"""

import asyncio, json, math, threading, time, websockets
from protocol import WEBSOCKET_PORT, MESSAGE_TYPES

GRAD_RATE     = 0.08
LATERAL_GAP_G = 5.0

def grad(cur, tgt, rate=GRAD_RATE):
    return cur + (tgt - cur) * rate

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get_formation_offsets(count):
    G = LATERAL_GAP_G
    if count == 1:   return [(0.0,  0.0)]
    elif count == 2: return [(0.0, -G),  (0.0,  G)]
    elif count == 3: return [(G,   0.0), (-G,  -G), (-G,  G)]
    elif count == 4: return [(G,   0.0), (0.0, -G), (0.0, G), (-G, 0.0)]
    elif count == 5: return [(G,   0.0), (0.0, -G), (0.0, G), (-G, -G*1.5), (-G, G*1.5)]
    else:
        r = G * 1.5
        return [(math.cos(2*math.pi*i/count - math.pi/2)*r,
                 math.sin(2*math.pi*i/count - math.pi/2)*r) for i in range(count)]


class FormationController:
    """
    Navigates formation CENTER using Pure Pursuit.
    Small lookahead (3 grid units) = reaches waypoints cleanly, no orbiting.
    All vehicles just read center + apply their fixed offset.
    """
    def __init__(self):
        self._lock      = threading.Lock()
        self.base_x     = 500.0
        self.base_depth = 500.0
        self.tangent    = 0.0
        self.waypoints  = []
        self.current_wp = 0
        self.active     = False
        self.speed      = 20.0
        self.running    = True
        threading.Thread(target=self._run, daemon=True).start()
        print("  🎯 FormationController started")

    def set_waypoints(self, wps):
        with self._lock:
            self.waypoints  = wps
            self.current_wp = 0
            self.active     = False
            if wps:
                self.base_x     = float(wps[0][0])
                self.base_depth = float(wps[0][1])
                if len(wps) > 1:
                    dx = float(wps[1][0]) - float(wps[0][0])
                    dy = float(wps[1][1]) - float(wps[0][1])
                    gx = (dx / 1000.0) * 100.0
                    gy = (dy / 6000.0) * 88.0
                    gd = math.sqrt(gx**2 + gy**2)
                    self.tangent = math.atan2(gy, gx) if gd > 0.001 else 0.0
                else:
                    self.tangent = 0.0
            print(f"  📍 Formation ready X={self.base_x:.0f}m D={self.base_depth:.0f}m "
                  f"hdg={math.degrees(self.tangent):.1f}°")

    def start(self, speed):
        with self._lock:
            self.speed = float(speed)
            if self.waypoints and not self.active:
                self.active     = True
                self.current_wp = 1 if len(self.waypoints) > 1 else 0
                print(f"  🚀 Formation STARTED {speed} m/s | {len(self.waypoints)} WPs")

    def set_speed(self, speed):
        with self._lock:
            self.speed = float(speed)

    def reset(self):
        with self._lock:
            self.active     = False
            self.current_wp = 0
            self.waypoints  = []
            self.base_x     = 500.0
            self.base_depth = 500.0
            self.tangent    = 0.0
            print("  🔄 FormationController reset")

    def get_state(self):
        with self._lock:
            return (self.base_x, self.base_depth,
                    self.tangent, self.active,
                    self.current_wp, self.speed)

    def _get_lookahead_point(self):
        """
        Pure Pursuit — Coulter (CMU 1992) implementation.

        Finds goal point by intersecting a circle of radius LD
        centred on the vehicle with each path segment.
        The vehicle then steers toward this goal via a circular arc.
        That arc curvature IS Pure Pursuit.
        """
        if not self.waypoints or self.current_wp >= len(self.waypoints):
            return None

        vx_g = (self.base_x     / 1000.0) * 100.0
        vy_g = (self.base_depth / 6000.0) * 88.0

        # Dynamic lookahead — shrinks near waypoint to prevent overshoot
        wp   = self.waypoints[self.current_wp]
        tx_g = (float(wp[0]) / 1000.0) * 100.0
        ty_g = (float(wp[1]) / 6000.0) * 88.0
        gdist = math.sqrt((vx_g-tx_g)**2 + (vy_g-ty_g)**2)

        BASE_LD = 4.0; MIN_LD = 1.5
        LD = (MIN_LD + max(0.0, gdist/(BASE_LD*2)) * (BASE_LD-MIN_LD)
              if gdist < BASE_LD*2 else BASE_LD)

        seg_start = max(0, self.current_wp - 1)
        goal = None

        # Intersect lookahead circle with each path segment
        for i in range(seg_start, len(self.waypoints) - 1):
            ax = (float(self.waypoints[i][0])     / 1000.0) * 100.0
            ay = (float(self.waypoints[i][1])     / 6000.0) * 88.0
            bx = (float(self.waypoints[i+1][0])   / 1000.0) * 100.0
            by = (float(self.waypoints[i+1][1])   / 6000.0) * 88.0
            dx = bx - ax; dy = by - ay
            fx = ax - vx_g; fy = ay - vy_g
            a = dx*dx + dy*dy
            b = 2*(fx*dx + fy*dy)
            c = fx*fx + fy*fy - LD*LD
            disc = b*b - 4*a*c
            if disc < 0 or a < 0.0001:
                continue
            sq = math.sqrt(disc)
            t2 = (-b + sq) / (2*a)
            t1 = (-b - sq) / (2*a)
            for t in [t2, t1]:
                if 0.0 <= t <= 1.0:
                    goal = (ax + t*dx, ay + t*dy)
                    break
            if goal:
                break

        if goal is None:
            # Fallback: steer directly to current waypoint
            goal = (tx_g, ty_g)

        return (goal[0]/100.0*1000.0, goal[1]/88.0*6000.0)

    def _step(self):
        """
        Pure Pursuit step (Coulter CMU 1992):
        1. Find lookahead goal point on path (circle intersection)
        2. Compute curvature k = 2*lateral_error / LD^2
        3. Update heading by curvature * speed * dt  (arc motion)
        4. Move forward in current heading — NOT toward goal directly
        This gives smooth curved arcs through waypoints.
        """
        dt = 0.1
        with self._lock:
            if not self.active or self.current_wp >= len(self.waypoints):
                return

            wp     = self.waypoints[self.current_wp]
            tx     = float(wp[0]); tdepth = float(wp[1])
            dx_wp  = tx - self.base_x; dd_wp = tdepth - self.base_depth
            gx_wp  = (dx_wp / 1000.0) * 100.0
            gy_wp  = (dd_wp / 6000.0) * 88.0
            gdist  = math.sqrt(gx_wp**2 + gy_wp**2)

            # Step 1: get lookahead goal
            goal = self._get_lookahead_point()
            if goal:
                gx_la = ((goal[0] - self.base_x)    / 1000.0) * 100.0
                gy_la = ((goal[1] - self.base_depth) / 6000.0) * 88.0
                LD    = math.sqrt(gx_la**2 + gy_la**2)
            else:
                gx_la = gx_wp; gy_la = gy_wp
                LD    = gdist

            if LD > 0.01:
                # Step 2: lateral error = cross product of heading and goal vector
                # heading unit vector
                hx = math.cos(self.tangent)
                hy = math.sin(self.tangent)
                # signed lateral distance to goal (positive = goal is to right)
                lateral = hx * gy_la - hy * gx_la
                # Pure Pursuit curvature formula: k = 2 * lateral / LD^2
                curvature = 2.0 * lateral / (LD * LD)
                curvature = clamp(curvature, -2.0, 2.0)

                # Step 3: heading update = curvature * speed * dt
                gs = self.speed * 0.01
                self.tangent += curvature * gs * dt
                # Keep tangent in [-pi, pi]
                self.tangent = math.atan2(math.sin(self.tangent),
                                          math.cos(self.tangent))

                # Step 4: move FORWARD in current heading (arc motion)
                step = gs * dt
                if step >= gdist or gdist < 0.5:
                    self.base_x     = tx
                    self.base_depth = tdepth
                else:
                    self.base_x     += math.cos(self.tangent)*step*(1000.0/100.0)
                    self.base_depth += math.sin(self.tangent)*step*(6000.0/88.0)

            # Waypoint reached: radius OR passed check
            WP_RADIUS = 2.5
            passed    = False
            if self.current_wp > 0:
                pw   = self.waypoints[self.current_wp - 1]
                px_g = (float(pw[0]) / 1000.0) * 100.0
                py_g = (float(pw[1]) / 6000.0) * 88.0
                tx_g = (tx           / 1000.0) * 100.0
                ty_g = (tdepth       / 6000.0) * 88.0
                vx_g = (self.base_x     / 1000.0) * 100.0
                vy_g = (self.base_depth / 6000.0) * 88.0
                sdx  = tx_g - px_g; sdy = ty_g - py_g
                rdx  = vx_g - tx_g; rdy = vy_g - ty_g
                if sdx*rdx + sdy*rdy > 0: passed = True

            if gdist < WP_RADIUS or passed:
                print(f"  ✅ Formation WP{self.current_wp} reached "
                      f"({'passed' if passed else 'radius'})")
                self.current_wp += 1
                if self.current_wp >= len(self.waypoints):
                    self.active = False
                    print("  🏁 Formation MISSION COMPLETE!")

    def _run(self):
        while self.running:
            self._step()
            time.sleep(0.1)


# Single shared formation controller
fc = FormationController()


class Vehicle:
    def __init__(self, vehicle_id):
        self.vehicle_id      = vehicle_id
        self.x_position      = 500.0
        self.depth           = 500.0
        self.altitude        = 100.0
        self.lat = 0.15; self.lon = 1.20; self.height = 0.0
        self.phi = 0.0; self.theta = 0.0; self.psi = 0.0
        self.roll = 0.0; self.pitch = 0.0; self.heading = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.u  = 0.0; self.v  = 0.0; self.w  = 0.0
        self.p  = 0.0; self.q  = 0.0; self.r  = 0.0
        self._depth_offset = 0.0; self._depth_offset_target = 0.0
        self.along_offset_g  = 0.0
        self.across_offset_g = 0.0
        self.mission_active  = False
        self.current_wp      = 0
        self.waypoints       = []
        self.speed           = 20.0
        self.battery = 95.0; self.temperature = 4.0
        self.pressure = 201.0; self.o2_level = 20.5
        self._t = 0.0; self.real_mode = False
        self.running = True
        self._lock = threading.Lock()
        print(f"  ✅ Vehicle: {vehicle_id}")

    def set_formation_offset(self, along_g, across_g):
        with self._lock:
            self.along_offset_g  = along_g
            self.across_offset_g = across_g
            print(f"  📐 [{self.vehicle_id}] along={along_g:+.1f}gu across={across_g:+.1f}gu")

    def set_waypoints(self, waypoints, start_pos=False):
        with self._lock:
            self.waypoints      = waypoints
            self.current_wp     = 0
            self.mission_active = False

    def start_mission(self): pass

    def set_speed(self, speed):
        with self._lock: self.speed = float(speed)

    def reset(self):
        with self._lock:
            self.x_position = 500.0; self.depth = 500.0
            self._depth_offset = 0.0
            self.waypoints = []; self.current_wp = 0; self.mission_active = False
            self.vx = self.vy = self.vz = 0.0
            self.phi = self.theta = self.psi = 0.0
            self.heading = self.r = self._t = 0.0; self.battery = 95.0
            print(f"  🔄 [{self.vehicle_id}] Reset")

    def update_from_input(self, data):
        with self._lock:
            for f in ["x_position","depth","altitude","vx","vy","vz",
                      "u","v","w","p","q","r","lat","lon"]:
                if f in data: setattr(self, f, float(data[f]))
            if "phi"   in data: self.phi   = float(data["phi"])
            if "theta" in data: self.theta = float(data["theta"])
            if "psi"   in data: self.psi   = float(data["psi"])
            self._update_degrees(); self.real_mode = True

    def _update_degrees(self):
        self.roll    = math.degrees(self.phi)
        self.pitch   = math.degrees(self.theta)
        self.heading = (math.degrees(self.psi) + 360) % 360

    def _clamp_all(self):
        self.pitch   = clamp(self.pitch,  -90,  90)
        self.roll    = clamp(self.roll,  -180, 180)
        self.heading = self.heading % 360

    def _simulate(self):
        t = self._t

        # Read formation center
        base_x, base_depth, tangent, active, wp_idx, spd = fc.get_state()
        self.mission_active = active
        self.current_wp     = wp_idx
        self.speed          = spd

        # Apply fixed 2D offset rotated by formation tangent
        # This keeps the rigid shape intact through all turns
        perp   = tangent - math.pi / 2
        off_gx = (math.cos(tangent) * self.along_offset_g +
                  math.cos(perp)    * self.across_offset_g)
        off_gy = (math.sin(tangent) * self.along_offset_g +
                  math.sin(perp)    * self.across_offset_g)

        self.x_position = clamp(base_x     + off_gx * (1000.0 / 100.0), 10.0, 990.0)
        self.depth      = clamp(base_depth + off_gy * (6000.0 / 88.0),  10.0, 5990.0)

        # Body heading follows Pure Pursuit tangent from FormationController
        # Same alpha as fc._step() so body stays aligned with formation
        diff = tangent - self.psi
        if diff >  math.pi: diff -= 2 * math.pi
        if diff < -math.pi: diff += 2 * math.pi
        # Match fc curvature update — heading rotates as the arc dictates
        self.psi = math.atan2(math.sin(self.psi + diff * 0.3),
                              math.cos(self.psi + diff * 0.3))

        # Velocity aligned with heading
        if active:
            gs      = spd * 0.01
            self.vx = math.cos(self.psi) * gs * (1000.0 / 100.0)
            self.vz = math.sin(self.psi) * gs * (6000.0 / 88.0)
        else:
            self.vx = grad(self.vx, 0.0, 0.15)
            self.vz = grad(self.vz, 0.0, 0.15)
        self.vy = 0.0

        # Gentle roll oscillation
        self.phi   = grad(self.phi, 0.04 * math.sin(t * 0.4) if active else 0.0, 0.08)
        self.theta = grad(self.theta, 0.0, 0.08)

        # Depth oscillation
        self._depth_offset_target = 2.0 * math.sin(t / 15.0)
        self._depth_offset = clamp(
            grad(self._depth_offset, self._depth_offset_target, 0.02), -2.0, 2.0)

        self.p = 0.0; self.q = 0.0; self.r = 0.0
        self.u = grad(self.u, abs(self.vx), 0.1)
        self.v = 0.0; self.w = grad(self.w, self.vz, 0.1)
        self.lat      = 0.15 + self.x_position * 0.000001
        self.lon      = 1.20 + self.depth      * 0.000001
        self.altitude = max(0, 6000 - self.depth + 100)
        self.battery     = grad(self.battery,     max(0, self.battery - 0.002), 0.3)
        self.temperature = grad(self.temperature, 4.0 + (self.depth/6000)*2, 0.02)
        self.pressure    = grad(self.pressure,    1.0 + self.depth/10.0, 0.05)
        self.o2_level    = grad(self.o2_level,    20.5, 0.01)
        self._t += 0.1
        self._update_degrees(); self._clamp_all()

    def get_state(self):
        with self._lock:
            _, _, _, active, wp_idx, _ = fc.get_state()
            n_wps  = len(self.waypoints)
            status = ("DIVING" if active else
                      "MISSION_COMPLETE" if (n_wps > 0 and not active and wp_idx >= n_wps)
                      else "WAITING")
            spd = math.sqrt(self.vx**2 + self.vz**2)
            return {
                "vehicle_id":       self.vehicle_id,
                "mode":             "REAL" if self.real_mode else "SIM",
                "x_position":       round(self.x_position, 2),
                "y_position":       round(self.depth, 2),
                "depth":            round(self.depth + self._depth_offset, 2),
                "depth_absolute":   round(self.depth, 2),
                "depth_offset":     round(self._depth_offset, 3),
                "altitude":         round(self.altitude, 2),
                "alt":              round(self.altitude, 2),
                "along_offset_g":   round(self.along_offset_g,  2),
                "across_offset_g":  round(self.across_offset_g, 2),
                "heading":          round(self.heading, 2),
                "yaw":              round(self.heading, 2),
                "pitch":            round(self.pitch, 2),
                "roll":             round(self.roll,  2),
                "phi":              round(self.phi,   6),
                "theta":            round(self.theta, 6),
                "psi":              round(self.psi,   6),
                "lat":              round(self.lat,   6),
                "lon":              round(self.lon,   6),
                "height":           round(self.height, 2),
                "vx":               round(self.vx, 3),
                "vy":               round(self.vy, 3),
                "vz":               round(self.vz, 3),
                "u":                round(self.u,  3),
                "v":                round(self.v,  3),
                "w":                round(self.w,  3),
                "speed":            round(spd, 3),
                "current_speed":    round(self.speed, 1),
                "instant_speed":    round(spd, 3),
                "p":                round(self.p, 4),
                "q":                round(self.q, 4),
                "r":                round(self.r, 4),
                "yaw_rate_degs":    round(math.degrees(self.r), 2),
                "battery":          round(self.battery,     1),
                "temperature":      round(self.temperature, 2),
                "pressure":         round(self.pressure,    2),
                "o2_level":         round(self.o2_level,    2),
                "vehicle_status":   status,
                "current_waypoint": wp_idx,
                "total_waypoints":  n_wps,
                "mission_active":   active,
                "thruster_fr": "ON" if active else "OFF",
                "thruster_fl": "ON" if active else "OFF",
                "thruster_rl": "ON" if active else "OFF",
                "thruster_rr": "ON" if active else "OFF",
            }

    def run(self):
        while self.running:
            with self._lock:
                if not self.real_mode: self._simulate()
                else: self._clamp_all()
            time.sleep(0.1)
        print(f"  🛑 {self.vehicle_id} stopped")


# ── Vehicle Manager ───────────────────────────────────────────

vehicles       = {}
base_waypoints = []
manager_lock   = threading.Lock()


def launch_all(vehicle_ids, waypoints_metres):
    global base_waypoints
    with manager_lock:
        base_waypoints = waypoints_metres
        offsets = get_formation_offsets(len(vehicle_ids))
        print(f"\n  🚀 LAUNCHING {len(vehicle_ids)} vehicles — RIGID FORMATION")
        for i, (along, across) in enumerate(offsets):
            print(f"     V{i+1}: along={along:+.1f}gu across={across:+.1f}gu")
        for i, vid in enumerate(vehicle_ids):
            if vid not in vehicles:
                v = Vehicle(vid)
                threading.Thread(target=v.run, daemon=True).start()
                vehicles[vid] = v
            along, across = offsets[i]
            vehicles[vid].set_formation_offset(along, across)
            vehicles[vid].set_waypoints(waypoints_metres, start_pos=False)
        fc.set_waypoints(waypoints_metres)
        print(f"  ✅ Formation ready — waiting for START!\n")


def add_vehicle(vehicle_id):
    with manager_lock:
        if vehicle_id in vehicles: return vehicles[vehicle_id]
        v = Vehicle(vehicle_id)
        threading.Thread(target=v.run, daemon=True).start()
        vehicles[vehicle_id] = v
        vids    = list(vehicles.keys())
        offsets = get_formation_offsets(len(vids))
        for i, vid in enumerate(vids):
            along, across = offsets[i]
            vehicles[vid].set_formation_offset(along, across)
            if base_waypoints:
                vehicles[vid].set_waypoints(base_waypoints, start_pos=False)
        if base_waypoints:
            fc.set_waypoints(base_waypoints)
        return v


def set_base_waypoints(wps):
    global base_waypoints
    with manager_lock:
        base_waypoints = wps
        offsets = get_formation_offsets(len(vehicles))
        for i, (vid, v) in enumerate(vehicles.items()):
            along, across = offsets[i]
            v.set_formation_offset(along, across)
            v.set_waypoints(wps, start_pos=False)
        fc.set_waypoints(wps)


def get_all_states():
    with manager_lock:
        return [v.get_state() for v in vehicles.values()]


# ── WebSocket stream port 8000 ────────────────────────────────

async def handle_client(websocket):
    print(f"  🌐 Frontend: {websocket.remote_address}")

    async def recv():
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                cmd = msg.get("type", "")
                if cmd == "LAUNCH_ALL":
                    vids = msg.get("vehicles", [])
                    wps  = [(float(w[0]), float(w[1])) for w in msg.get("waypoints", [])]
                    print(f"  📡 LAUNCH_ALL: {vids} | {len(wps)} WPs")
                    launch_all(vids, wps)
                elif cmd == "ADD_VEHICLE":
                    add_vehicle(msg.get("vehicle_id", "Vehicle-1"))
                elif cmd == "SET_WAYPOINTS":
                    wps = [(float(w[0]), float(w[1])) for w in msg.get("waypoints", [])]
                    set_base_waypoints(wps)
                elif cmd == "SET_SPEED":
                    fc.set_speed(float(msg.get("speed", 20)))
                elif cmd == "START_MISSION":
                    spd = float(msg.get("speed", 20))
                    fc.start(spd)
                    print(f"  ▶ START_MISSION {spd} m/s")
                elif cmd == "UPDATE_VEHICLE":
                    vid  = msg.get("vehicle_id", "")
                    data = msg.get("data", {})
                    with manager_lock:
                        if vid not in vehicles: add_vehicle(vid)
                        vehicles[vid].update_from_input(data)
                elif cmd == "RESET":
                    global base_waypoints
                    fc.reset()
                    with manager_lock:
                        for v in vehicles.values(): v.reset()
                        vehicles.clear(); base_waypoints = []
            except Exception as e:
                print(f"  ⚠️ {e}")

    asyncio.create_task(recv())

    try:
        while True:
            states = get_all_states()
            if states:
                all_done = all(s["vehicle_status"] == "MISSION_COMPLETE" for s in states)
                await websocket.send(json.dumps({"type": "FULL_VEHICLE_STATE", "vehicles": states}))
                await websocket.send(json.dumps({
                    "type": "ORIENTATION_DATA",
                    "vehicles": [{"vehicle_id": s["vehicle_id"], "pitch": s["pitch"],
                                  "roll": s["roll"], "yaw": s["yaw"], "mode": s["mode"],
                                  "mission_active": s["mission_active"]} for s in states]
                }))
                await websocket.send(json.dumps({"type": MESSAGE_TYPES["SENSOR_DATA"], "data": states}))
                bx, bd, tang, active, wp_idx, spd = fc.get_state()
                print(f"  [FC] ({bx:.0f}m,{bd:.0f}m) hdg={math.degrees(tang):.1f}° "
                      f"WP={wp_idx}/{len(base_waypoints)} "
                      f"{'MOVING' if active else 'IDLE'} spd={spd:.0f}")
                for s in states:
                    print(f"    [{s['vehicle_id']:10s}] "
                          f"X={s['x_position']:6.0f}m D={s['depth_absolute']:6.0f}m "
                          f"Hdg={s['heading']:5.1f}°")
                await asyncio.sleep(1.0 if all_done else 0.1)
            else:
                print("  ⏳ Waiting..."); await asyncio.sleep(0.5)
            print("-" * 75)
    except websockets.exceptions.ConnectionClosed:
        print("  Frontend disconnected")


# ── WebSocket commands port 8001 ──────────────────────────────

async def handle_commands(websocket):
    print(f"  🔌 AUV/cmd: {websocket.remote_address}")
    try:
        async for raw in websocket:
            msg = json.loads(raw)
            cmd = msg.get("type", "")
            if cmd == "LAUNCH_ALL":
                vids = msg.get("vehicles", [])
                wps  = [(float(w[0]), float(w[1])) for w in msg.get("waypoints", [])]
                launch_all(vids, wps)
                await websocket.send(json.dumps({"type": "ACK", "message": f"Launched {len(vids)}"}))
            elif cmd == "SET_WAYPOINTS":
                wps = [(float(w[0]), float(w[1])) for w in msg.get("waypoints", [])]
                set_base_waypoints(wps)
                await websocket.send(json.dumps({"type": "ACK", "message": f"{len(wps)} WPs"}))
            elif cmd == "SET_SPEED":
                spd = float(msg.get("speed", 20))
                fc.set_speed(spd)
                await websocket.send(json.dumps({"type": "ACK", "message": f"Speed {spd}"}))
            elif cmd == "START_MISSION":
                spd = float(msg.get("speed", 20))
                fc.start(spd)
                await websocket.send(json.dumps({"type": "ACK", "message": f"Started {spd} m/s"}))
            elif cmd == "ADD_VEHICLE":
                add_vehicle(msg.get("vehicle_id", "Vehicle-1"))
                await websocket.send(json.dumps({"type": "ACK"}))
            elif cmd == "RESET":
                global base_waypoints
                fc.reset()
                with manager_lock:
                    for v in vehicles.values(): v.reset()
                    vehicles.clear(); base_waypoints = []
                await websocket.send(json.dumps({"type": "ACK", "message": "Reset"}))
            elif cmd == "GET_ALL_VEHICLES":
                await websocket.send(json.dumps({"type": "FULL_VEHICLE_STATE", "vehicles": get_all_states()}))
            elif cmd == "UPDATE_VEHICLE":
                vid  = msg.get("vehicle_id", "")
                data = msg.get("data", {})
                with manager_lock:
                    if vid not in vehicles: add_vehicle(vid)
                    vehicles[vid].update_from_input(data)
                await websocket.send(json.dumps({"type": "ACK"}))
    except websockets.exceptions.ConnectionClosed:
        print("  AUV/cmd disconnected")


async def main():
    print("=" * 75)
    print("  UNDERWATER VEHICLE — FORMATION NAVIGATION SERVER")
    print("  NIOT Deep Sea Technology Dept")
    print("=" * 75)
    print(f"  Stream   : ws://0.0.0.0:{WEBSOCKET_PORT}  (frontend)")
    print(f"  Commands : ws://0.0.0.0:8001              (AUVs)")
    print("=" * 75)
    print("=" * 75)
    async with (
        websockets.serve(handle_client,   "0.0.0.0", WEBSOCKET_PORT),
        websockets.serve(handle_commands, "0.0.0.0", 8001),
    ):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())