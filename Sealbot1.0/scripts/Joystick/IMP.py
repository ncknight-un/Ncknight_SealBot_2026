# source venv/bin/activate
# pip install websocket-client
import websocket
import json
import time
import threading

WS_URI = "ws://192.168.2.2/mavlink2rest/ws/mavlink"

def on_message(ws, message):
    timestamp = time.time()

    # Print raw messages for debugging
    print(f"[RAW {timestamp:.3f}]: {message}")

    try:
        msg = json.loads(message)
    except json.JSONDecodeError:
        return  # ignore non-JSON messages

    mtype = msg.get('msg', {}).get('name')
    data = msg.get('msg', {}).get('fields', {})

    if mtype == "MANUAL_CONTROL":
        print(f"{timestamp:.3f} JOYSTICK: x={data.get('x')}, y={data.get('y')}, "
              f"z={data.get('z')}, r={data.get('r')}, buttons={data.get('buttons')}")
    elif mtype == "ACTUATOR_OUTPUT_STATUS":
        actuators = data.get('actuator', [])
        print(f"{timestamp:.3f} MOTORS: {actuators}")

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

    # Send subscription after a short delay to ensure socket is ready
    def subscribe():
        time.sleep(0.1)
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "subscribe",
            "params": {
                "messages": ["MANUAL_CONTROL", "ACTUATOR_OUTPUT_STATUS"]
            },
            "id": 1
        }
        ws.send(json.dumps(subscribe_msg))
        print("Subscription sent")

    threading.Thread(target=subscribe).start()

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WS_URI,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()



# import websocket
# import json
# import time

# WS_URI = "ws://192.168.2.2/mavlink2rest/ws/mavlink"

# def on_message(ws, message):
#     timestamp = time.time()

#     try:
#         msg = json.loads(message)
#     except json.JSONDecodeError:
#         # Ignore non-JSON messages (like empty pings)
#         return

#     mtype = msg.get('msg', {}).get('name')
#     data = msg.get('msg', {}).get('fields', {})

#     if mtype == "MANUAL_CONTROL":
#         print(f"{timestamp:.3f} JOYSTICK: x={data.get('x')}, y={data.get('y')}, "
#               f"z={data.get('z')}, r={data.get('r')}, buttons={data.get('buttons')}")
#     elif mtype == "ACTUATOR_OUTPUT_STATUS":
#         actuators = data.get('actuator', [])
#         print(f"{timestamp:.3f} MOTORS: {actuators}")

# def on_error(ws, error):
#     print("WebSocket error:", error)

# def on_close(ws, close_status_code, close_msg):
#     print("WebSocket closed")

# def on_open(ws):
#     print("WebSocket connection opened")
#     # Subscribe to the messages we care about
#     subscribe_msg = {
#         "jsonrpc": "2.0",
#         "method": "subscribe",
#         "params": {
#             "messages": ["MANUAL_CONTROL", "ACTUATOR_OUTPUT_STATUS"]
#         },
#         "id": 1
#     }
#     ws.send(json.dumps(subscribe_msg))

# if __name__ == "__main__":
#     ws = websocket.WebSocketApp(
#         WS_URI,
#         on_open=on_open,
#         on_message=on_message,
#         on_error=on_error,
#         on_close=on_close
#     )
#     ws.run_forever()












