import socket
import threading
import time
import webbrowser
from contextlib import closing
from errno import EADDRINUSE

from web_panel import create_server


HOST = "127.0.0.1"
PORT = 8080
MAX_PORT_SCAN = 20


def wait_for_server(host, port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.3)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def port_is_open(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def resolve_server_port(host, preferred_port):
    try:
        return create_server(host, preferred_port), preferred_port, False
    except OSError as exc:
        if exc.errno != EADDRINUSE:
            raise

    if port_is_open(host, preferred_port):
        return None, preferred_port, True

    for port in range(preferred_port + 1, preferred_port + MAX_PORT_SCAN + 1):
        try:
            return create_server(host, port), port, False
        except OSError as exc:
            if exc.errno != EADDRINUSE:
                raise

    raise RuntimeError(
        f"Nenhuma porta livre encontrada entre {preferred_port} e "
        f"{preferred_port + MAX_PORT_SCAN}."
    )


def main():
    server, port, reused_existing = resolve_server_port(HOST, PORT)
    server_thread = None
    if server is not None:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

    url = f"http://{HOST}:{port}"
    if reused_existing:
        print(f"Painel web ja estava ativo em {url}")
    else:
        print(f"Painel web iniciado em {url}")
    print("O navegador sera aberto para capturar a camera com permissao do browser.")

    if wait_for_server(HOST, port):
        webbrowser.open(url, new=1)
    else:
        print("O servidor demorou para responder. Abra a URL manualmente se necessario.")

    try:
        while server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nEncerrando painel...")
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
