import sys
import msvcrt
from dynamixel_sdk import *  # pip install dynamixel-sdk

# ---------------------
# PARAMÈTRES DYNAMIXEL
# ---------------------
DEVICENAME = 'COM10'
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0
DXL_ID = 11

# ---------------------
# ADRESSES EX-106+
# ---------------------
ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# ---------------------
# PARAMÈTRES MÉCANIQUES
# ---------------------
PAS_VIS_MM = 25.4
DISTANCE_MAX_MM = 130
TOURS_MAX = DISTANCE_MAX_MM / PAS_VIS_MM  # ≈ 5.12 tours
ANGLE_MAX = TOURS_MAX * 360               # ≈ 1843.2°
STEP_ANGLE = 10                           # angle par touche
ANGLE_TO_POS = 4095 / 360

# ---------------------
# OUTILS CLAVIER WINDOWS
# ---------------------
def getch():
    return msvcrt.getch().decode('utf-8')

def angle_to_position(angle_deg):
    return max(0, min(4095, int(angle_deg * ANGLE_TO_POS)))

def position_to_angle(position_val):
    return position_val * 360 / 4095

# ---------------------
# INITIALISATION DYNAMIXEL
# ---------------------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Échec ouverture du port.")
    sys.exit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Échec réglage baudrate.")
    sys.exit()

# Activer le couple
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)

# Lire position initiale
dxl_present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
current_angle = position_to_angle(dxl_present_position)
print(f"Position initiale : {current_angle:.2f}°")

# ---------------------
# BOUCLE DE CONTRÔLE CLAVIER
# ---------------------
print("\nCommandes : q = avancer, d = reculer, x = quitter\n")

while True:
    key = getch()
    if key == 'q':
        if current_angle + STEP_ANGLE <= ANGLE_MAX:
            current_angle += STEP_ANGLE
            pos = angle_to_position(current_angle)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, pos)
            print(f"→ Avance à {current_angle:.2f}°")
        else:
            print("⚠️ Limite avant atteinte.")
    elif key == 'd':
        if current_angle - STEP_ANGLE >= 0:
            current_angle -= STEP_ANGLE
            pos = angle_to_position(current_angle)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, pos)
            print(f"← Recule à {current_angle:.2f}°")
        else:
            print("⚠️ Limite arrière atteinte.")
    elif key == 'x':
        print("Fermeture...")
        break

# Désactivation du couple et fermeture port
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()
