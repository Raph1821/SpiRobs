from dynamixel_sdk import *  # SDK classes
import time

# ---------------------
# PARAMÈTRES À ADAPTER (SUR DYNAMIXEL WIZARD 2.0)
# ---------------------
DEVICENAME = 'COM10'           # Port série (ex: COM3 sous Windows, /dev/ttyUSB0 sous Linux)
BAUDRATE = 1000000             # Vitesse (essaie aussi 1000000 si ça ne marche pas)
PROTOCOL_VERSION = 1.0         # EX-106+ utilise le protocole 1.0
DXL_ID = 11                    # ID de ton moteur (à adapter si différent)

# ---------------------
# REGISTRES DYNAMIXEL (EX-106+) 
# Ces lignes définissent les adresses mémoire internes du moteur. Chaque moteur Dynamixel
# possède une mémoire divisée en registres auxquels on accède pour lire ou écrire des données.
# ---------------------
ADDR_MX_TORQUE_ENABLE = 24     # Activer couple
ADDR_MX_GOAL_POSITION = 30     # Position cible
ADDR_MX_PRESENT_POSITION = 36  # Position actuelle
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# ---------------------
# INITIALISATION
# ---------------------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Ouvrir le port
if not portHandler.openPort():
    print("Échec ouverture du port.")
    quit()

# Régler le débit
if not portHandler.setBaudRate(BAUDRATE):
    print("Échec réglage baudrate.")
    quit()

# Ping le servo
dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, DXL_ID)
if dxl_comm_result != COMM_SUCCESS:
    print("Erreur communication :", packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("Erreur du moteur :", packetHandler.getRxPacketError(dxl_error))
else:
    print(f"Moteur détecté ! Modèle : {dxl_model_number}")

# Activer le moteur
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)

# Aller à une position (valeurs entre 0 et 4095 pour EX-106+)
goal_position = 0  # Position centrale
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_position)
print("Position envoyée : ", goal_position)

time.sleep(1)

# Lire la position réelle
present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
print("Position actuelle : ", present_position)

time.sleep(1)

# Aller à une position (valeurs entre 0 et 4095 pour EX-106+)
goal_position = 4095  # Position centrale
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_position)
print("Position envoyée : ", goal_position)

time.sleep(1)

# Lire la position réelle
present_position, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
print("Position actuelle : ", present_position)

# Désactiver le couple à la fin
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)

# Fermer le port
portHandler.closePort()
