import time
from dynamixel_sdk import *  # SDK Dynamixel

# Paramètres à modifier
overture = False    # Ouverture de la pince ou fermeture
desired_rps = 0.6   # Vitesse désirée en tours par seconde
distance_tours = 3  # Nombre de tours à parcourir

# Config port
DEVICENAME = 'COM10'
BAUDRATE = 57600    # Vitesse de transmission entre le contrôleur et le moteur

# ID moteur
DXL_ID = 1

# Registres MX-106 (protocole 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Conversion vitesse
# 1 unité = 0.114 rpm (approx) => vitesse valeur = rpm / 0.114
VELOCITY_UNIT = 0.114   # 1 unité = 0,114 tr/min d'après le site constructeur
                        # Ex : Si la valeur est 300 unités, cela correspond à environ 300*0.114 = 34,2 tr/min

# Calcul temps pour X tours
time_to_run = distance_tours / desired_rps
print(f"Temps de rotation : {time_to_run}")

# Initialisation port
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(2.0)

if not portHandler.openPort():
    print("Impossible d'ouvrir le port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Impossible de définir le baudrate")
    quit()

# Activation du wheel mode (rotation infinie)

ADDR_CW_LIMIT = 6   # limite dans le sens horaire
ADDR_CCW_LIMIT = 8  # limite dans le sens anti-horaire

# Mettre les limites à 0 pour permettre la rotation continue (désactiver les bornes)
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CW_LIMIT, 0)
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CCW_LIMIT, 0)

# Activation torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque activé")

# Calcul vitesse en rpm puis valeur moteur
desired_rpm = desired_rps * 60
dxl_velocity_value = int(desired_rpm / VELOCITY_UNIT)   # Nb d'unités * 0.114 = tr/min

if dxl_velocity_value > 1024:
    raise ValueError("Vitesse du moteur trop grande !")

if overture :                       # Le moteur tourne en sens CCW (anti-horaire) pour des valeurs de 1 à 1023, et s’arrête si la valeur est 0.
    dxl_velocity_value += 1024      # Il tourne en sens CW (horaire) pour des valeurs de 1025 à 2047, et s’arrête si la valeur est 1024.

print(f"Vitesse souhaitée: {desired_rps} tours/s => {desired_rpm} tours/min => valeur = {dxl_velocity_value}")

# Envoi vitesse
dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, dxl_velocity_value)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

print(f"Moteur tourne durant {time_to_run:.2f} secondes...")
time.sleep(time_to_run)

# Arrêt moteur
dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, 0)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

# Désactivation torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque désactivé")

portHandler.closePort()
print("Programme terminé.")
