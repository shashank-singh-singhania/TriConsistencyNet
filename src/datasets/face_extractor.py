"""
TriConsistencyNet

Face Extraction Pipeline

Author: Shashank Singh
"""

from pathlib import Path

import cv2
from retinaface import RetinaFace
from tqdm import tqdm

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


class FaceExtractor:

    def __init__(self):

        config = ConfigLoader().load("dataset.yaml")

        self.frames_root = (
            Path(config.dataset.processed)
            / "frames"
        )

        self.faces_root = (
            Path(config.dataset.processed)
            / "faces"
        )

        self.image_size = config.dataset.image_size

    def extract_video(self, manipulation, video_name):

        input_dir = (
            self.frames_root
            / manipulation
            / video_name
        )

        output_dir = (
            self.faces_root
            / manipulation
            / video_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if any(output_dir.iterdir()):
            project_logger.info(
                f"Skipping {manipulation}/{video_name}"
            )
            return

        image_paths = sorted(
            input_dir.glob("*.png")
        )

        saved = 0

        for image_path in image_paths:

            image = cv2.imread(str(image_path))

            faces = RetinaFace.detect_faces(image)

            if not isinstance(faces, dict):
                continue

            largest = None
            largest_area = 0

            for _, face in faces.items():

                x1, y1, x2, y2 = face["facial_area"]

                area = (x2 - x1) * (y2 - y1)

                if area > largest_area:

                    largest_area = area

                    largest = (x1, y1, x2, y2)

            if largest is None:
                continue

            x1, y1, x2, y2 = largest

            crop = image[y1:y2, x1:x2]

            crop = cv2.resize(
                crop,
                (
                    self.image_size,
                    self.image_size,
                ),
            )

            output_path = (
                output_dir
                / image_path.name
            )

            cv2.imwrite(
                str(output_path),
                crop,
            )

            saved += 1

        project_logger.success(
            f"{manipulation}/{video_name} : {saved} faces"
        )

    def extract_from_split(self, split_csv: str, workers=None):

        import pandas as pd

        dataframe = pd.read_csv(split_csv)

        tasks = []
        for _, row in dataframe.iterrows():
            relative_path = Path(row["File Path"])
            manipulation = relative_path.parts[0]
            video_name = relative_path.stem
            tasks.append((manipulation, video_name))

        if workers is not None and workers > 1:
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            num_workers = min(workers, 16)
            project_logger.info(f"Extracting faces using {num_workers} parallel workers on GPU...")
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_faces_worker,
            ) as pool:
                list(tqdm(
                    pool.imap_unordered(_extract_faces_worker, tasks),
                    total=len(tasks),
                    desc="Face Extraction",
                ))
        else:
            project_logger.info("Extracting faces sequentially...")
            for manipulation, video_name in tqdm(tasks, desc="Face Extraction"):
                self.extract_video(manipulation, video_name)


def _init_faces_worker():
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    global _worker_extractor
    from src.datasets.face_extractor import FaceExtractor
    _worker_extractor = FaceExtractor()


def _extract_faces_worker(task):
    manipulation, video_name = task
    global _worker_extractor
    _worker_extractor.extract_video(manipulation, video_name)