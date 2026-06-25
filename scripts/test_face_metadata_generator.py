from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.face_metadata_generator import FaceMetadataGenerator


def main():

    generator = FaceMetadataGenerator()

    dataframe = generator.generate()

    print()

    print(dataframe.head())

    print()

    print("Total Faces:", len(dataframe))


if __name__ == "__main__":

    main()