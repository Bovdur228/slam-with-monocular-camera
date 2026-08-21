import os
import sys

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# =========================================================================================================================================
# DBoW3 bindings loading
# =========================================================================================================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_BINDINGS_DIR = (
    _PROJECT_ROOT
    / "cpp"
    / "build"
    / "Release"
)


_dll_directory_handles = []


def _load_dbow3_bindings():
    """
    Загружает скомпилированный pybind11-модуль DBoW3.

    C++ build artifacts находятся в:
        cpp/build/Release/

    Функция ничего не компилирует.
    Она только делает уже собранный модуль доступным Python.
    """

    if not _BINDINGS_DIR.exists():
        raise RuntimeError(
            "DBoW3 bindings build directory does not exist: "
            f"{_BINDINGS_DIR}"
        )

    if os.name == "nt":

        # os.add_dll_directory() возвращает handle.
        # Его нужно сохранять живым, пока DLL используются.
        dll_handle = os.add_dll_directory(
            str(_BINDINGS_DIR)
        )

        _dll_directory_handles.append(
            dll_handle
        )

    bindings_path = str(
        _BINDINGS_DIR
    )

    if bindings_path not in sys.path:
        sys.path.insert(
            0,
            bindings_path
        )

    try:
        import dbow3_bindings

    except ImportError as exc:

        raise RuntimeError(
            "Failed to import dbow3_bindings. "
            "Build the C++ bindings first."
        ) from exc

    return dbow3_bindings


dbow3_bindings = _load_dbow3_bindings()


# =========================================================================================================================================
# Query result
# =========================================================================================================================================

@dataclass(frozen=True)
class BoWQueryResult:
    """
    Один результат поиска в BoW database.

    keyframe_id:
        ID KeyFrame в нашей SLAM map.

    entry_id:
        внутренний ID записи DBoW3 database.

    score:
        visual similarity score DBoW3.
        Чем больше score, тем визуально более похожи KeyFrames.
    """

    keyframe_id: int
    entry_id: int
    score: float


# =========================================================================================================================================
# BoW Database
# =========================================================================================================================================

class BoWDatabase:
    """
    Python-слой над DBoW3 Vocabulary и Database.

    Отвечает только за place-recognition infrastructure:

        ORB descriptors
            ↓
        Vocabulary
            ↓
        BoW representation
            ↓
        DBoW3 database
            ↓
        similarity query

    Класс НЕ занимается:
        - temporal filtering;
        - covisibility filtering;
        - loop consistency;
        - geometric verification;
        - loop correction.

    Всё это будет ответственностью LoopCandidateDetector.
    """

    def __init__(
        self
    ):
        self.vocabulary = dbow3_bindings.Vocabulary()

        self.database = None

        # DBoW3 использует собственные sequential entry IDs:
        #
        # 0, 1, 2, 3, ...
        #
        # Они НЕ обязаны совпадать с KeyFrame.id.
        self.entry_to_keyframe_id = {}

        self.keyframe_to_entry_id = {}


    # -------------------------------------------------------------------------------------------------------------------------------------
    def create_vocabulary(
        self,
        descriptor_sets,
        branching_factor=10,
        depth=5
    ):
        """
        Создаёт visual vocabulary из набора ORB descriptor matrices.

        descriptor_sets:
            iterable из np.ndarray shape (N, 32), dtype uint8.

        После создания нового vocabulary старая database очищается,
        потому что database всегда должна соответствовать конкретному
        vocabulary.
        """

        valid_descriptor_sets = []

        for descriptors in descriptor_sets:

            descriptors = self._validate_descriptors(
                descriptors,
                allow_empty=True
            )

            if len(descriptors) == 0:
                continue

            valid_descriptor_sets.append(
                descriptors
            )

        if len(valid_descriptor_sets) == 0:
            raise ValueError(
                "No valid descriptor sets were provided "
                "for vocabulary creation."
            )

        if branching_factor <= 1:
            raise ValueError(
                "branching_factor must be greater than 1."
            )

        if depth <= 0:
            raise ValueError(
                "depth must be greater than 0."
            )

        self.vocabulary.create(
            valid_descriptor_sets,
            branching_factor=int(branching_factor),
            depth=int(depth)
        )

        if self.vocabulary.size() == 0:
            raise RuntimeError(
                "DBoW3 created an empty vocabulary."
            )

        self._reset_database()


    # -------------------------------------------------------------------------------------------------------------------------------------
    def load_vocabulary(
        self,
        path
    ):
        """
        Загружает заранее сохранённый vocabulary с диска.

        После загрузки database создаётся заново.
        """

        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Vocabulary file does not exist: {path}"
            )

        success = self.vocabulary.load(
            str(path)
        )

        if not success:
            raise RuntimeError(
                f"Failed to load vocabulary: {path}"
            )

        self._reset_database()


    # -------------------------------------------------------------------------------------------------------------------------------------
    def save_vocabulary(
        self,
        path
    ):
        """
        Сохраняет текущий vocabulary.
        """

        self._require_vocabulary()

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vocabulary.save(
            str(path)
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def add_keyframe(
        self,
        keyframe
    ):
        """
        Добавляет KeyFrame в BoW database.

        Использует:
            keyframe.id
            keyframe.des

        Возвращает:
            DBoW3 entry_id.
        """

        if keyframe is None:
            raise ValueError(
                "keyframe cannot be None."
            )

        if keyframe.id is None:
            raise ValueError(
                "KeyFrame must have a valid id."
            )

        if not hasattr(
            keyframe,
            "des"
        ):
            raise ValueError(
                "KeyFrame does not contain descriptors."
            )

        return self.add(
            keyframe_id=int(keyframe.id),
            descriptors=keyframe.des
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def add(
        self,
        keyframe_id,
        descriptors
    ):
        """
        Добавляет ORB descriptors одного KeyFrame в database.

        keyframe_id:
            ID из нашей SLAM map.

        descriptors:
            np.ndarray shape (N, 32), dtype uint8.
        """

        self._require_database()

        keyframe_id = int(
            keyframe_id
        )

        if keyframe_id in self.keyframe_to_entry_id:
            raise ValueError(
                f"KeyFrame {keyframe_id} is already present "
                "in the BoW database."
            )

        descriptors = self._validate_descriptors(
            descriptors
        )

        entry_id = int(
            self.database.add(
                descriptors
            )
        )

        self.entry_to_keyframe_id[
            entry_id
        ] = keyframe_id

        self.keyframe_to_entry_id[
            keyframe_id
        ] = entry_id

        return entry_id


    # -------------------------------------------------------------------------------------------------------------------------------------
    def query(
        self,
        descriptors,
        max_results=10
    ):
        """
        Ищет наиболее визуально похожие KeyFrames.

        Возвращает:
            list[BoWQueryResult]

        Результат уже содержит реальные KeyFrame IDs,
        а не только внутренние DBoW3 entry IDs.
        """

        self._require_database()

        if max_results <= 0:
            return []

        descriptors = self._validate_descriptors(
            descriptors
        )

        raw_results = self.database.query(
            descriptors,
            max_results=int(max_results)
        )

        results = []

        for entry_id, score in raw_results:

            entry_id = int(
                entry_id
            )

            keyframe_id = self.entry_to_keyframe_id.get(
                entry_id
            )

            if keyframe_id is None:
                continue

            results.append(
                BoWQueryResult(
                    keyframe_id=int(keyframe_id),
                    entry_id=entry_id,
                    score=float(score)
                )
            )

        return results


    # -------------------------------------------------------------------------------------------------------------------------------------
    def query_keyframe(
        self,
        keyframe,
        max_results=10
    ):
        """
        Convenience wrapper для query() по KeyFrame.
        """

        if keyframe is None:
            raise ValueError(
                "keyframe cannot be None."
            )

        if not hasattr(
            keyframe,
            "des"
        ):
            raise ValueError(
                "KeyFrame does not contain descriptors."
            )

        return self.query(
            descriptors=keyframe.des,
            max_results=max_results
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def score(
        self,
        descriptors_a,
        descriptors_b
    ):
        """
        Считает DBoW3 similarity score между двумя
        наборами ORB descriptors.
        """

        self._require_vocabulary()

        descriptors_a = self._validate_descriptors(
            descriptors_a
        )

        descriptors_b = self._validate_descriptors(
            descriptors_b
        )

        return float(
            self.vocabulary.score(
                descriptors_a,
                descriptors_b
            )
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def contains_keyframe(
        self,
        keyframe_id
    ):
        """
        Проверяет, добавлен ли KeyFrame в database.
        """

        return int(
            keyframe_id
        ) in self.keyframe_to_entry_id


    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_entry_id(
        self,
        keyframe_id
    ):
        """
        Возвращает DBoW3 entry ID для SLAM KeyFrame ID.

        Если KF отсутствует:
            возвращает None.
        """

        return self.keyframe_to_entry_id.get(
            int(keyframe_id)
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_keyframe_id(
        self,
        entry_id
    ):
        """
        Обратное отображение:
            DBoW3 entry ID -> SLAM KeyFrame ID.
        """

        return self.entry_to_keyframe_id.get(
            int(entry_id)
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def vocabulary_size(
        self
    ):
        """
        Количество visual words.
        """

        return int(
            self.vocabulary.size()
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def size(
        self
    ):
        """
        Количество KeyFrames в BoW database.
        """

        if self.database is None:
            return 0

        return int(
            self.database.size()
        )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def get_stats(
        self
    ):
        """
        Возвращает базовую диагностику BoW database.
        """

        return {
            "vocabulary_size": self.vocabulary_size(),
            "database_size": self.size(),
            "mapped_entries": len(self.entry_to_keyframe_id),
            "mapped_keyframes": len(self.keyframe_to_entry_id)
        }


    # =====================================================================================================================================
    # Internal helpers
    # =====================================================================================================================================

    def _reset_database(
        self
    ):
        """
        Создаёт новую пустую DBoW3 database
        для текущего vocabulary.
        """

        self._require_vocabulary()

        self.database = dbow3_bindings.Database(
            self.vocabulary
        )

        self.entry_to_keyframe_id.clear()
        self.keyframe_to_entry_id.clear()


    # -------------------------------------------------------------------------------------------------------------------------------------
    def _require_vocabulary(
        self
    ):
        """
        Проверяет, что vocabulary уже создан или загружен.
        """

        if self.vocabulary.size() == 0:
            raise RuntimeError(
                "Vocabulary has not been created or loaded yet."
            )


    # -------------------------------------------------------------------------------------------------------------------------------------
    def _require_database(
        self
    ):
        """
        Проверяет наличие initialized DBoW3 database.
        """

        if self.database is None:
            raise RuntimeError(
                "BoW database is not initialized. "
                "Create or load a vocabulary first."
            )


    # -------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _validate_descriptors(
        descriptors,
        allow_empty=False
    ):
        """
        Проверяет формат ORB descriptors.

        Ожидается:
            np.ndarray
            shape = (N, 32)
            dtype = uint8
        """

        if descriptors is None:

            if allow_empty:
                return np.empty(
                    (0, 32),
                    dtype=np.uint8
                )

            raise ValueError(
                "Descriptors cannot be None."
            )

        descriptors = np.asarray(
            descriptors
        )

        if descriptors.ndim != 2:
            raise ValueError(
                "ORB descriptors must be a 2D array."
            )

        if descriptors.shape[1] != 32:
            raise ValueError(
                "ORB descriptors must have shape (N, 32). "
                f"Received: {descriptors.shape}"
            )

        if len(descriptors) == 0 and not allow_empty:
            raise ValueError(
                "ORB descriptors cannot be empty."
            )

        if descriptors.dtype != np.uint8:
            descriptors = descriptors.astype(
                np.uint8,
                copy=False
            )

        # pybind11 wrapper ожидает contiguous memory.
        descriptors = np.ascontiguousarray(
            descriptors
        )

        return descriptors