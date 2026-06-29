"""
TriConsistencyNet

Frame Extraction Pipeline

Author: Shashank Singh
"""

from pathlib import Path

import cv2
from tqdm import tqdm

from src.utils.config import ConfigLoader
from src.utils.logger import project_logger


class FrameExtractor:

    def __init__(self):

        config = ConfigLoader().load("dataset.yaml")

        self.dataset_root = Path(config.dataset.root)

        self.processed_root = Path(config.dataset.processed)

        self.target_fps = config.dataset.fps

    def extract_video(
        self,
        relative_video_path: str,
    ):

        video_path = self.dataset_root / relative_video_path

        manipulation = Path(relative_video_path).parts[0]

        video_name = Path(relative_video_path).stem

        output_directory = (

            self.processed_root

            / "frames"

            / manipulation

            / video_name

        )

        output_directory.mkdir(

            parents=True,

            exist_ok=True,

        )

        if any(output_directory.iterdir()):

            return

        capture = cv2.VideoCapture(

            str(video_path)

        )

        original_fps = capture.get(

            cv2.CAP_PROP_FPS

        )

        frame_interval = max(

            int(original_fps / self.target_fps),

            1,

        )

        frame_index = 0

        saved_index = 0

        while True:

            success, frame = capture.read()

            if not success:

                break

            if frame_index % frame_interval == 0:

                output_path = (

                    output_directory

                    / f"{saved_index:06d}.png"

                )

                # Use compression level 1 (fastest lossless PNG encoding, 5-10x faster than default)
                cv2.imwrite(

                    str(output_path),

                    frame,

                    [cv2.IMWRITE_PNG_COMPRESSION, 1],

                )

                saved_index += 1

            frame_index += 1

        capture.release()

    def extract_from_dataframe(

        self,

        dataframe,

        limit=None,

        workers=None,

    ):

        rows = dataframe

        if limit:

            rows = dataframe.head(limit)

        video_paths = rows["File Path"].tolist()

        if workers is not None and workers > 1:

            import multiprocessing

            num_workers = min(workers, 256)

            project_logger.info(f"Extracting frames using {num_workers} parallel workers...")

            with multiprocessing.Pool(
                processes=num_workers,
                initializer=_init_worker,
            ) as pool:

                list(tqdm(

                    pool.imap_unordered(_extract_video_worker, video_paths),

                    total=len(video_paths),

                    desc="Frame Extraction",

                ))

        else:

            project_logger.info("Extracting frames sequentially...")

            for video_path in tqdm(video_paths, desc="Frame Extraction"):

                self.extract_video(video_path)

    def extract_from_split(self, split_csv: str, workers=None):

        import pandas as pd

        dataframe = pd.read_csv(split_csv)

        self.extract_from_dataframe(dataframe, workers=workers)


def _init_worker():

    global _worker_extractor

    _worker_extractor = FrameExtractor()


def _extract_video_worker(relative_video_path: str):

    global _worker_extractor

    _worker_extractor.extract_video(relative_video_path)