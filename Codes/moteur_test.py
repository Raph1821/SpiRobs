import time
from dynamixel_sdk import *  # SDK Dynamixel

# Paramètres
overture = False
desired_rps = 0.6
distance_tours = 3

DEVICENAME = 'COM10'
BAUDRATE = 57600
DXL_ID = 1

# Registres protocole 1.0
ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_VELOCITY = 32
ADDR_CW_LIMIT = 6
ADDR_CCW_LIMIT = 8

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
VELOCITY_UNIT = 0.114

time_to_run = distance_tours / desired_rps
print(f"Temps de rotation : {time_to_run:.2f} secondes")

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(1.0)   # Protocole 1.0

if not portHandler.openPort():
    print("Impossible d'ouvrir le port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Impossible de définir le baudrate")
    quit()

# Wheel mode
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CW_LIMIT, 0)
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CCW_LIMIT, 0)

# Torque ON
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque activé")

# Calcul vitesse
desired_rpm = desired_rps * 60
dxl_velocity_value = int(desired_rpm / VELOCITY_UNIT)
if dxl_velocity_value > 1023:
    raise ValueError("Vitesse trop grande !")

if not overture:
    dxl_velocity_value |= 1024   # bit 10 = sens CW

print(f"Vitesse souhaitée: {desired_rpm:.2f} rpm => valeur = {dxl_velocity_value}")

# ⚠️ Ecriture 2 octets, pas 4
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_VELOCITY, dxl_velocity_value)
if dxl_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(packetHandler.getRxPacketError(dxl_error))

print(f"Moteur tourne {time_to_run:.2f} sec...")
time.sleep(time_to_run)

# Stop moteur
dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_VELOCITY, 0)

# Torque OFF
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
if dxl_comm_result != COMM_SUCCESS:
    print(packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print(packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque désactivé")

portHandler.closePort()
print("Programme terminé.")
