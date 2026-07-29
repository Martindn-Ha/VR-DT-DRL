# How to run on the real UR3e arm

This guide uses **two computers**:

| Machine | Job |
|---------|-----|
| **Windows PC** | Runs the AI / vision program (`gpu_server.py`) |
| **Ubuntu VM** | Talks to the real robot, gripper, and camera |

You do **not** need Webots for the real arm.

**What happens each try:** camera picture → find the block → grasp policy → arm picks it up. You place blocks on the board by hand.

Lab defaults below use robot IP `192.168.1.120`. Change IPs if your lab differs.

Finish [one-time setup](setup.md) on Windows first.

---

## Big picture (do this in order)

1. Fix network (robot ↔ VM, and VM ↔ Windows AI).
2. Copy latest client code onto the VM.
3. Start robot driver → enable External Control on the robot teach pendant.
4. Start gripper node, then **reset and activate** it.
5. Start camera.
6. Start AI on Windows.
7. Start the client on the VM (last).

Use a **separate terminal** for each VM program. In every VM terminal, run the “Source ROS” commands first.

---

## Step 1 — Robot ↔ VM network

The robot must call back to the **VM’s IP on the robot network**.

On the VM, after every reboot:

```bash
ip route get 192.168.1.120 | grep src
```

Take the address after `src` (example: `192.168.1.148`). Use it in **three** places:

1. Robot teach pendant → **External Control** → Host IP = that `src`
2. Port = `50002`
3. Driver command `reverse_ip:=` = that same `src` (see Step 4)

Do not reuse an old IP from memory. After reboot the VM may switch network cards.

If two cards both have a `192.168.1.x` address, keep one and turn the other off:

```bash
ip -br addr | grep 192.168.1
sudo ip link set ens37 down
```

(Only disable the card that is **not** the one reaching the robot.)

---

## Step 2 — VM ↔ Windows AI network

On the VM, edit `vm_simulation_system/config/network_config.yaml`:

- `host_ip` = Windows PC address the VM can reach (often `192.168.241.1` on VMware NAT)
- `host_port` = `8888`

Do **not** use `127.0.0.1` here — that is only for simulation on one machine.

On Windows, `host_gpu_system/config/network_config.yaml` should use `host_ip: "0.0.0.0"` so it accepts connections.

---

## Step 3 — Copy client code to the VM

On the VM:

```bash
bash /mnt/hgfs/VR-DT-DRL/scripts/sync_vm_sim_client.sh
```

If you see `$'\r': command not found`, the script has Windows line endings — convert it to Linux endings, then run again.

If `/mnt/hgfs/VR-DT-DRL` is missing:

```bash
sudo mkdir -p /mnt/hgfs/VR-DT-DRL
sudo vmhgfs-fuse .host:/VR-DT-DRL /mnt/hgfs/VR-DT-DRL \
  -o allow_other,uid=$(id -u),gid=$(id -g)
```

---

## Step 4 — Source ROS (every new VM terminal)

```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/install_isolated/setup.bash
```

---

## Step 5 — Start the robot driver

Replace `192.168.1.148` with **your** `src` from Step 1.

```bash
roslaunch ur_robot_driver ur3e_bringup.launch \
  robot_ip:=192.168.1.120 \
  reverse_ip:=192.168.1.148 \
  use_tool_communication:=true \
  tool_voltage:=24 \
  tool_device_name:=/tmp/ttyUR \
  kinematics_config:=/home/seth/calibration.yaml
```

On the robot teach pendant: open **External Control** and press **Play once**.

Wait about 30 seconds. You want a stable connection (not a repeating “Connection dropped” message).

---

## Step 6 — Start the gripper

Only after `/tmp/ttyUR` exists:

```bash
ls -la /tmp/ttyUR
```

```bash
ROS_NAMESPACE=ur3e_robot1 rosrun robotiq_2f_gripper_control \
  Robotiq2FGripperRtuNode.py /tmp/ttyUR __name:=gripper_node
```

### Reset and activate (required)

Use a **new terminal** (source ROS first). Stream with `-r 10` (not a one-shot publish).

**1. Reset / deactivate** (~2 seconds, then Ctrl+C):

```bash
rostopic pub -r 10 /ur3e_robot1/Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output "rACT: 0
rGTO: 0
rATR: 0
rPR: 0
rSP: 0
rFR: 0"
```

**2. Activate** (~3 seconds, then Ctrl+C):

```bash
rostopic pub -r 10 /ur3e_robot1/Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output "rACT: 1
rGTO: 1
rATR: 0
rPR: 0
rSP: 255
rFR: 150"
```

**3. Check it is ready:**

```bash
rostopic echo /ur3e_robot1/Robotiq2FGripperRobotInput -n 1
```

You want `gFLT: 0` and `gSTA: 3`. If `gFLT` is not 0, reset and activate again.

Optional open/close test (after ready):

```bash
# Close
rostopic pub -r 10 /ur3e_robot1/Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output "rACT: 1
rGTO: 1
rATR: 0
rPR: 255
rSP: 255
rFR: 150"

# Open
rostopic pub -r 10 /ur3e_robot1/Robotiq2FGripperRobotOutput robotiq_2f_gripper_control/Robotiq2FGripper_robot_output "rACT: 1
rGTO: 1
rATR: 0
rPR: 0
rSP: 255
rFR: 150"
```

---

## Step 7 — Start the camera

```bash
roslaunch realsense2_camera rs_camera.launch \
  align_depth:=true initial_reset:=true \
  color_width:=640 color_height:=360 color_fps:=15 \
  depth_width:=640 depth_height:=360 depth_fps:=15
```

Optional check:

```bash
rostopic hz /camera/color/image_raw
```

---

## Step 8 — Start the AI on Windows

### Normal picking (no typing)

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py --local-bbox-dqn
```

### Pick by typed instruction (optional)

Needs Ollama on Windows with `qwen3-vl:8b-instruct`.

```powershell
cd host_gpu_system
.\venv\Scripts\Activate.ps1
python src\gpu_server.py --local-bbox-dqn --use-vlm-select --vlm-model qwen3-vl:8b-instruct
```

---

## Step 9 — Start the client on the VM (last)

### Normal picking

```bash
cd ~/catkin_ws/src/vm_simulation_system
python3 src/simulation_client.py --ros-camera --robot-id 1 --use-local-bbox-dqn --free
```

### Pick by typed instruction

Use the matching AI command from Step 8 (with `--use-vlm-select`).

```bash
cd ~/catkin_ws/src/vm_simulation_system
python3 src/simulation_client.py --ros-camera --robot-id 1 --use-local-bbox-dqn --use-vlm-select --free
```

1. Place blocks on the physical board.
2. Open `host_gpu_system/debug/board_warp_r1_latest.jpg` on Windows when prompted.
3. Type the instruction in the VM client terminal (for example `yellow bottom right`).

`--free` means you place the block; the software does not spawn it.

---

## Where to put the block (R1)

Rough usable area on the board (world meters): X about `-0.790` to `-0.502`, Z about `0.757` to `0.925` (center near `-0.646`, `0.841`).

---

## Quick health checks (VM)

```bash
ip route get 192.168.1.120 | grep src
ping -c 3 192.168.1.120
pgrep -af ur_robot_driver
sudo ss -tlnp | grep -E '50001|50002'
ls -la /tmp/ttyUR
rostopic hz /camera/color/image_raw
```

---

## If something fails

| What you see | What to try |
|--------------|-------------|
| Pendant “connection refused” | Driver not running, or Host IP / port `50002` wrong |
| “Connection dropped” looping | Kill old drivers; re-check `src` IP; only one robot-network card; start client later |
| Launch file not found | Run Step 4 (source ROS) again |
| Gripper topics up but no motion | `gFLT` / `gSTA` wrong — run reset + activate again |
| `/tmp/ttyUR` missing | Fix driver + External Control Play first |
| Robot “compile error” | Power-cycle the robot controller; restart driver with the correct `reverse_ip` |

Reset stuck drivers:

```bash
pkill -f ur_robot_driver; pkill -f roslaunch; sleep 2
```
