import base64
import json
import mimetypes
import platform
import subprocess
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
WEB_DIR = ROOT_DIR / "web"


class BrowserMemeMatcher:
    def __init__(self, assets_folder="assets"):
        self.assets_path = Path(assets_folder)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

        self.memes = []
        self.meme_profiles = {
            "angry_baby": {
                "label": "Angry Baby",
                "smile": 0.05,
                "eyes": 0.08,
                "contrast": 0.58,
                "brightness": 0.42,
            },
            "disaster_girl": {
                "label": "Disaster Girl",
                "smile": 0.18,
                "eyes": 0.11,
                "contrast": 0.52,
                "brightness": 0.48,
            },
            "gene_wilder": {
                "label": "Gene Wilder",
                "smile": 0.16,
                "eyes": 0.09,
                "contrast": 0.45,
                "brightness": 0.5,
            },
            "leonardo_dicaprio": {
                "label": "Leonardo Dicaprio",
                "smile": 0.5,
                "eyes": 0.1,
                "contrast": 0.5,
                "brightness": 0.56,
            },
            "overly_attached_girlfriend": {
                "label": "Overly Attached Girlfriend",
                "smile": 0.28,
                "eyes": 0.18,
                "contrast": 0.62,
                "brightness": 0.56,
            },
            "success_kid": {
                "label": "Success Kid",
                "smile": 0.4,
                "eyes": 0.1,
                "contrast": 0.64,
                "brightness": 0.47,
            },
        }
        self.load_memes()

    def load_memes(self):
        image_files = (
            list(self.assets_path.glob("*.jpg"))
            + list(self.assets_path.glob("*.png"))
            + list(self.assets_path.glob("*.jpeg"))
        )
        for img_file in sorted(image_files):
            slug = img_file.stem.lower()
            profile = self.meme_profiles.get(slug)
            if profile is None:
                continue

            self.memes.append(
                {
                    "name": profile["label"],
                    "slug": slug,
                    "path": str(img_file),
                    "url": f"/assets/{img_file.name}",
                    "profile": profile,
                }
            )
            print(f"Loaded: {img_file.name}")

        print(f"\nTotal memes loaded: {len(self.memes)}")

    def extract_face_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(120, 120),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = gray[y : y + h, x : x + w]
        upper_roi = face_roi[: max(h // 2, 1), :]

        eyes = self.eye_cascade.detectMultiScale(
            upper_roi,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(18, 18),
        )
        smiles = self.smile_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.7,
            minNeighbors=24,
            minSize=(35, 20),
        )

        eye_areas = [ew * eh for (_, _, ew, eh) in eyes[:2]]
        avg_eye_area = (
            (sum(eye_areas) / len(eye_areas)) / float(w * h)
            if eye_areas
            else 0.0
        )
        smile_score = 0.0
        for _, sy, sw, sh in smiles:
            # Smiles that appear lower on the face tend to be more useful.
            vertical_bias = (sy + sh / 2) / max(h, 1)
            score = (sw / max(w, 1)) * vertical_bias
            smile_score = max(smile_score, score)

        brightness = float(np.mean(face_roi) / 255.0)
        contrast = float(np.std(face_roi) / 128.0)

        return {
            "smile": min(smile_score, 1.0),
            "eyes": min(avg_eye_area * 18.0, 1.0),
            "contrast": min(contrast, 1.0),
            "brightness": min(brightness, 1.0),
            "face_box": [int(x), int(y), int(w), int(h)],
        }

    def find_best_match(self, user_features):
        if user_features is None:
            return None, 0.0

        best_meme = None
        best_score = -1.0
        for meme in self.memes:
            profile = meme["profile"]
            score = self.compute_similarity(user_features, profile)
            if score > best_score:
                best_score = score
                best_meme = meme

        return best_meme, best_score

    def compute_similarity(self, features, profile):
        weights = {
            "smile": 0.4,
            "eyes": 0.28,
            "contrast": 0.18,
            "brightness": 0.14,
        }

        score = 0.0
        for key, weight in weights.items():
            diff = abs(features[key] - profile[key])
            score += weight * max(0.0, 1.0 - diff * 1.8)

        return round(score * 100.0, 2)


class MatcherService:
    def __init__(self, assets_folder="assets"):
        self.matcher = BrowserMemeMatcher(assets_folder=assets_folder)
        self.lock = threading.Lock()

    def list_memes(self):
        return [
            {
                "name": meme["name"],
                "path": meme["path"],
                "url": meme["url"],
            }
            for meme in self.matcher.memes
        ]

    def match_data_url(self, data_url):
        if "," not in data_url:
            return {"status": "error", "message": "Imagem invalida."}

        _, encoded = data_url.split(",", 1)
        try:
            image_bytes = base64.b64decode(encoded)
        except ValueError:
            return {"status": "error", "message": "Falha ao decodificar imagem."}

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return {"status": "error", "message": "Nao foi possivel ler o frame."}

        with self.lock:
            user_features = self.matcher.extract_face_features(frame)
            best_meme, score = self.matcher.find_best_match(user_features)

        if best_meme is None:
            return {
                "status": "no-face",
                "message": "Nenhum rosto detectado.",
                "score": None,
                "match": None,
            }

        return {
            "status": "ok",
            "message": "Rosto detectado.",
            "score": score,
            "match": {
                "name": best_meme["name"],
                "path": best_meme["path"],
                "url": best_meme["url"],
            },
            "features": {
                "smile": round(user_features["smile"], 3),
                "eyes": round(user_features["eyes"], 3),
            },
        }


SERVICE = MatcherService(assets_folder=str(ASSETS_DIR))


class MemePanelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.serve_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/health":
            return self.send_json({"status": "ok"})
        if path == "/api/capabilities":
            return self.send_json(get_capabilities(client_safe=True))
        if path == "/api/memes":
            return self.send_json({"memes": SERVICE.list_memes()})
        if path.startswith("/assets/"):
            filename = Path(unquote(path.removeprefix("/assets/"))).name
            return self.serve_file(ASSETS_DIR / filename)

        self.send_error(HTTPStatus.NOT_FOUND, "Rota nao encontrada.")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/open-photo-booth":
            return self.open_photo_booth()
        if parsed.path != "/api/match":
            self.send_error(HTTPStatus.NOT_FOUND, "Rota nao encontrada.")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json(
                {"status": "error", "message": "JSON invalido."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        image_data = data.get("image")
        if not image_data:
            self.send_json(
                {"status": "error", "message": "Campo image ausente."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        response = SERVICE.match_data_url(image_data)
        status = HTTPStatus.OK if response["status"] != "error" else HTTPStatus.BAD_REQUEST
        self.send_json(response, status=status)

    def open_photo_booth(self):
        capabilities = get_capabilities()
        photo_app = capabilities["photo_app"]
        if not photo_app["supported"]:
            self.send_json(
                {
                    "status": "error",
                    "message": photo_app["message"],
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        try:
            subprocess.run(
                photo_app["command"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            self.send_json(
                {
                    "status": "error",
                    "message": f"Nao foi possivel abrir {photo_app['label']}.",
                    "detail": str(exc).strip(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self.send_json(
            {
                "status": "ok",
                "message": f"{photo_app['label']} aberto. Tire uma foto e envie o arquivo para o painel.",
            }
        )

    def serve_file(self, file_path, content_type=None):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Arquivo nao encontrado.")
            return

        guessed_type = content_type or mimetypes.guess_type(file_path.name)[0]
        mime_type = guessed_type or "application/octet-stream"
        content = file_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


def run_server():
    server = create_server()
    host, port = server.server_address
    print(f"Painel web disponivel em http://{host}:{port}")
    print("Abra a rota no navegador e permita o acesso a camera.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def create_server():
    return ThreadingHTTPServer(("localhost", 0), MemePanelHandler)


def get_capabilities(client_safe=False):
    system = platform.system().lower()
    photo_app = {
        "supported": system == "darwin",
        "label": "Photo Booth",
        "command": ["open", "-a", "Photo Booth"] if system == "darwin" else None,
        "message": "" if system == "darwin" else "Abrir Photo Booth pelo painel so e suportado no macOS.",
    }

    if client_safe:
        photo_app = {
            "supported": photo_app["supported"],
            "label": photo_app["label"],
            "message": photo_app["message"],
        }

    return {"photo_app": photo_app}


if __name__ == "__main__":
    run_server()
