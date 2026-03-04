import socket
import threading
import time
import webbrowser
from contextlib import closing

from web_panel import create_server


def wait_for_server(host, port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def main():
    server = create_server()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"Painel web iniciado em {url}")
    print("O navegador sera aberto para capturar a camera com permissao do browser.")

    if wait_for_server(host, port):
        webbrowser.open(url, new=1)
    else:
        print("O servidor demorou para responder. Abra a URL manualmente se necessario.")

    try:
        while server_thread.is_alive():
            server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nEncerrando painel...")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
