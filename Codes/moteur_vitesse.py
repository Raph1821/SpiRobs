import time
from dynamixel_sdk import *  # SDK Dynamixel

# Paramètres à modifier
sens_horaire = 1    # Ouverture de la pince ou fermeture
desired_rps = 0.5   # Vitesse désirée en tours par seconde
distance_tours = 3  # Nombre de tours à parcourir

# Config port
DEVICENAME = 'COM10'
BAUDRATE = 57600    # Vitesse de transmission entre le contrôleur et le moteur

# ID moteur
DXL_ID = 1

# Registres MX-106 (protocole 1.0 !)
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_VELOCITY = 32
ADDR_PRESENT_POSITION = 36

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

# Conversion vitesse
VELOCITY_UNIT = 0.114   # 1 unité = 0,114 tr/min

# Calcul temps pour X tours
time_to_run = distance_tours / desired_rps
print(f"Temps de rotation : {time_to_run}")

# Initialisation port
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(1.0)

if not portHandler.openPort():
    print("Impossible d'ouvrir le port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Impossible de définir le baudrate")
    quit()

# Activation du wheel mode (rotation infinie)
ADDR_CW_LIMIT = 6
ADDR_CCW_LIMIT = 8
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
dxl_velocity_value = int(desired_rpm / VELOCITY_UNIT)

if dxl_velocity_value > 1024:
    raise ValueError("Vitesse du moteur trop grande !")

if sens_horaire:
    dxl_velocity_value += 1024  # Sens horaire

print(f"Vitesse souhaitée: {desired_rps} tours/s => {desired_rpm} tours/min => valeur = {dxl_velocity_value}")

# Lecture position initiale
prev_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
if dxl_comm_result != COMM_SUCCESS:
    print("Erreur lecture position:", packetHandler.getTxRxResult(dxl_comm_result))
    quit()
elif dxl_error != 0:
    print("Erreur lecture position:", packetHandler.getRxPacketError(dxl_error))
    quit()

initial_position = prev_position
turn_count = 0

# Envoi vitesse
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, dxl_velocity_value)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

print(f"Moteur tourne durant {time_to_run:.2f} secondes...")

start_time = time.time()
while time.time() - start_time < time_to_run:
    current_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("Erreur lecture position:", packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("Erreur lecture position:", packetHandler.getRxPacketError(dxl_error))
    else:
        # Détection dépassement tour
        delta = current_position - prev_position

        if delta < -2048:
            turn_count += 1
        elif delta > 2048:
            turn_count -= 1

        # Position totale en unités
        total_position_units = (turn_count * 4096) + current_position - initial_position

        # Conversion en tours
        total_tours = total_position_units / 4096

        print(f"Position cumulée: {total_tours:.3f} tours")

        prev_position = current_position

    time.sleep(0.01)

# Lecture position finale
final_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
if dxl_comm_result != COMM_SUCCESS:
    print("Erreur lecture position:", packetHandler.getTxRxResult(dxl_comm_result))
    quit()
elif dxl_error != 0:
    print("Erreur lecture position:", packetHandler.getRxPacketError(dxl_error))
    quit()

# Arrêt moteur
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, 0)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))

# Remettre les limites (réactiver Joint Mode et PID)
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CW_LIMIT, 0)
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CCW_LIMIT, 4095)

# Commande de position
ADDR_GOAL_POSITION = 30
target_position = final_position   # Arrêter où il est
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_position)

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