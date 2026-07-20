#!/bin/bash
# Copy latest vm_simulation_system Python modules from VMware share to catkin_ws.

SHARE="/mnt/hgfs/VR-DT-DRL/vm_simulation_system/src"
DEST="$HOME/catkin_ws/src/vm_simulation_system/src"

if [ ! -d "$SHARE" ]; then
  sudo mkdir -p /mnt/hgfs/VR-DT-DRL
  sudo vmhgfs-fuse .host:/VR-DT-DRL /mnt/hgfs/VR-DT-DRL \
    -o allow_other,uid="$(id -u)",gid="$(id -g)" || true
fi

if [ ! -f "$SHARE/simulation_client.py" ]; then
  echo "ERROR: share not mounted at $SHARE" >&2
  exit 1
fi

cp "$SHARE"/*.py "$DEST/"
echo "Synced Python files to $DEST"
python3 "$DEST/simulation_client.py" --help | grep use-geo-grasp && echo "OK: --use-geo-grasp available"
