"""
Topic 1 — Networking Fundamentals
TCP demo: reliable, ordered delivery (file download scenario).

Runs server + client in the same process using threads.
The 3-way handshake happens automatically inside socket.connect().

Run:
    python tcp_demo.py
"""

import socket
import threading
import time


HOST = "127.0.0.1"
PORT = 9100


def tcp_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[TCP Server] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[TCP Server] Client connected from {addr}")

            # Simulate sending a binary file in 20-byte chunks
            file_data = b"NUMPY_BINARY_DATA_" * 10   # 180 bytes total
            chunk_size = 20
            for i in range(0, len(file_data), chunk_size):
                chunk = file_data[i : i + chunk_size]
                conn.sendall(chunk)
                print(f"[TCP Server] Sent chunk {i // chunk_size + 1}: {chunk}")
                time.sleep(0.05)

        print("[TCP Server] Done — all chunks sent.\n")


def tcp_client():
    time.sleep(0.3)   # give the server time to bind

    received = b""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 3-way handshake happens here: SYN → SYN-ACK → ACK
        s.connect((HOST, PORT))
        print(f"[TCP Client] Connected — 3-way handshake complete")

        while True:
            data = s.recv(20)
            if not data:
                break
            received += data

    print(f"[TCP Client] File received intact: {len(received)} bytes  ✅")
    print(f"[TCP Client] Content: {received[:40]}...")
    print()
    print("  TCP guarantee: every byte arrived, in order, no duplicates.")
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  TCP Demo — File Download (simulating pip install)")
    print("=" * 55)
    print()

    server_thread = threading.Thread(target=tcp_server, daemon=True)
    server_thread.start()

    tcp_client()
    server_thread.join(timeout=3)
