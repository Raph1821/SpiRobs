import sys
import msvcrt
from dynamixel_sdk import *  # pip install dynamixel-sdk
from numpy import pi

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
TOURS_MAX = DISTANCE_MAX_MM / PAS_VIS_MM * 2*pi  # ≈ 5.12 tours

# ---------------------
# OUTILS CLAVIER WINDOWS
# ---------------------
def getch():
    return msvcrt.getch().decode('utf-8')

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
print(f"Position initiale : {dxl_present_position:.2f}°")

# ---------------------
# BOUCLE DE CONTRÔLE CLAVIER
# ---------------------
print("\nCommandes : o = ouvert, f = fermé, x = quitter\n")

while True:
    key = getch()
    dxl_present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
    if key == 'o':
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, 0)
        print(f"→ Avance à {dxl_present_position:.2f}°")
    elif key == 'f':
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, TOURS_MAX)
        print(f"← Recule à {dxl_present_position:.2f}°")
    elif key == 'x':
        print("Annulation...")
        break

# Désactivation du couple et fermeture port
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()
