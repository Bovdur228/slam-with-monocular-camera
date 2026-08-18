class CovisibilityGraph:
    """
    Граф совместной видимости KeyFrames.

    Вершины:
        KeyFrames.

    Ребро между двумя KeyFrames существует,
    если они наблюдают достаточное количество общих MapPoints.

    Вес ребра:
        количество общих MapPoints.

    graph имеет структуру:

        {
            kf_id: {
                other_kf_id: shared_mappoint_count,
                ...
            },
            ...
        }

    Граф симметричный:

        graph[A][B] == graph[B][A]
    """

    def __init__(
        self,
        min_shared_mappoints=15
    ):
        """
        Создаёт пустой Covisibility Graph.

        min_shared_mappoints:
            минимальное количество общих MapPoints,
            необходимое для создания edge между двумя KeyFrames.
        """

        self.min_shared_mappoints = int(
            min_shared_mappoints
        )

        # kf_id -> {other_kf_id: weight}
        self.graph = {}

        # kf_id -> KeyFrame object
        self.keyframes = {}


    # ---------------------------------------------------------------------------------------------------------------------------------
    def add_keyframe(
        self,
        keyframe,
        update_connections=True
    ):
        """
        Добавляет KeyFrame в граф.

        Если update_connections=True,
        сразу вычисляет его связи с другими KeyFrames.
        """

        if keyframe is None:
            return False

        if keyframe.id is None:
            return False

        kf_id = int(
            keyframe.id
        )

        self.keyframes[kf_id] = keyframe

        if kf_id not in self.graph:
            self.graph[kf_id] = {}

        if update_connections:
            self.update_keyframe(
                keyframe
            )

        return True


    # ---------------------------------------------------------------------------------------------------------------------------------
    def remove_keyframe(
        self,
        keyframe_or_id
    ):
        """
        Полностью удаляет KeyFrame и все связанные с ним edges.

        Принимает:
            либо объект KeyFrame,
            либо его id.

        Возвращает:
            True, если KeyFrame существовал в графе.
        """

        kf_id = self._get_keyframe_id(
            keyframe_or_id
        )

        if kf_id is None:
            return False

        if kf_id not in self.graph:
            self.keyframes.pop(
                kf_id,
                None
            )

            return False

        self._remove_all_edges_for_keyframe(
            kf_id
        )

        self.graph.pop(
            kf_id,
            None
        )

        self.keyframes.pop(
            kf_id,
            None
        )

        return True


    # ---------------------------------------------------------------------------------------------------------------------------------
    def update_keyframe(
        self,
        keyframe
    ):
        """
        Полностью пересчитывает covisibility connections
        для одного KeyFrame.

        Алгоритм:
            1. Удаляет его старые edges.
            2. Проходит по MapPoints текущего KeyFrame.
            3. Через MapPoint.observations находит другие KeyFrames,
               наблюдающие те же MapPoints.
            4. Считает число общих MapPoints.
            5. Создаёт симметричные edges для достаточно сильных связей.

        Возвращает:
            словарь {other_kf_id: weight}
            только для активных edges.
        """

        if keyframe is None:
            return {}

        if keyframe.id is None:
            return {}

        kf_id = int(
            keyframe.id
        )

        # На случай, если update вызывается до add_keyframe().
        self.keyframes[kf_id] = keyframe

        if kf_id not in self.graph:
            self.graph[kf_id] = {}

        # Старые веса могли перестать быть актуальными
        # после triangulation / culling / outlier rejection.
        self._remove_all_edges_for_keyframe(
            kf_id
        )

        connection_counts = {}

        # Защита от возможных duplicate MapPoints
        # внутри keyframe.map_points.
        seen_mappoint_ids = set()

        for map_point in keyframe.map_points:

            if map_point is None:
                continue

            if map_point.id is None:
                continue

            mp_id = int(
                map_point.id
            )

            if mp_id in seen_mappoint_ids:
                continue

            seen_mappoint_ids.add(
                mp_id
            )

            # Один и тот же другой KF должен учитываться
            # максимум один раз для данной MapPoint.
            seen_other_keyframe_ids = set()

            for observation in map_point.observations:

                if observation is None:
                    continue

                other_keyframe = observation.keyframe

                if other_keyframe is None:
                    continue

                if other_keyframe.id is None:
                    continue

                other_kf_id = int(
                    other_keyframe.id
                )

                # KeyFrame не может быть covisible сам с собой.
                if other_kf_id == kf_id:
                    continue

                if other_kf_id in seen_other_keyframe_ids:
                    continue

                seen_other_keyframe_ids.add(
                    other_kf_id
                )

                # Сохраняем объект другого KeyFrame,
                # если его ещё не было в registry.
                self.keyframes[other_kf_id] = other_keyframe

                if other_kf_id not in self.graph:
                    self.graph[other_kf_id] = {}

                connection_counts[other_kf_id] = (
                    connection_counts.get(
                        other_kf_id,
                        0
                    )
                    + 1
                )

        active_connections = {}

        for other_kf_id, weight in connection_counts.items():

            if weight < self.min_shared_mappoints:
                continue

            self._set_edge(
                kf_id,
                other_kf_id,
                weight
            )

            active_connections[other_kf_id] = weight

        return active_connections


    # ---------------------------------------------------------------------------------------------------------------------------------
    def rebuild(
        self,
        keyframes
    ):
        """
        Полностью перестраивает Covisibility Graph с нуля.

        Полезно:
            - после серьёзных изменений карты;
            - после MapPoint culling;
            - для diagnostics;
            - для проверки incremental update;
            - при будущем loading карты.

        Возвращает:
            stats графа после rebuild.
        """

        self.graph = {}
        self.keyframes = {}

        valid_keyframes = []

        for keyframe in keyframes:

            if keyframe is None:
                continue

            if keyframe.id is None:
                continue

            kf_id = int(
                keyframe.id
            )

            self.keyframes[kf_id] = keyframe
            self.graph[kf_id] = {}

            valid_keyframes.append(
                keyframe
            )

        for keyframe in valid_keyframes:
            self.update_keyframe(
                keyframe
            )

        return self.get_stats()


    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_weight(
        self,
        keyframe_a,
        keyframe_b
    ):
        """
        Возвращает вес edge между двумя KeyFrames.

        Если edge отсутствует:
            возвращает 0.
        """

        kf_a_id = self._get_keyframe_id(
            keyframe_a
        )

        kf_b_id = self._get_keyframe_id(
            keyframe_b
        )

        if kf_a_id is None or kf_b_id is None:
            return 0

        return int(
            self.graph.get(
                kf_a_id,
                {}
            ).get(
                kf_b_id,
                0
            )
        )


    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_neighbors(
        self,
        keyframe,
        min_weight=None
    ):
        """
        Возвращает всех covisible neighbours KeyFrame.

        Формат:
            [
                (KeyFrame, weight),
                ...
            ]

        Результат сортируется по убыванию weight.
        """

        kf_id = self._get_keyframe_id(
            keyframe
        )

        if kf_id is None:
            return []

        if min_weight is None:
            min_weight = self.min_shared_mappoints

        neighbors = []

        for other_kf_id, weight in self.graph.get(
            kf_id,
            {}
        ).items():

            if weight < min_weight:
                continue

            other_keyframe = self.keyframes.get(
                other_kf_id
            )

            if other_keyframe is None:
                continue

            neighbors.append(
                (
                    other_keyframe,
                    int(weight)
                )
            )

        neighbors.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return neighbors


    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_best_neighbors(
        self,
        keyframe,
        max_neighbors=10,
        min_weight=None
    ):
        """
        Возвращает самые сильные covisible KeyFrames.

        Например:
            [
                (KF20, 180),
                (KF18, 140),
                (KF21, 95)
            ]
        """

        neighbors = self.get_neighbors(
            keyframe,
            min_weight=min_weight
        )

        if max_neighbors is None:
            return neighbors

        if max_neighbors <= 0:
            return []

        return neighbors[:max_neighbors]


    # ---------------------------------------------------------------------------------------------------------------------------------
    def are_covisible(
        self,
        keyframe_a,
        keyframe_b,
        min_weight=None
    ):
        """
        Проверяет, являются ли два KeyFrames достаточно covisible.
        """

        if min_weight is None:
            min_weight = self.min_shared_mappoints

        weight = self.get_weight(
            keyframe_a,
            keyframe_b
        )

        return weight >= min_weight


    # ---------------------------------------------------------------------------------------------------------------------------------
    def get_stats(
        self
    ):
        """
        Возвращает базовую статистику Covisibility Graph.
        """

        num_keyframes = len(
            self.graph
        )

        # Graph симметричный, поэтому каждое edge
        # хранится два раза:
        #
        # A -> B
        # B -> A
        total_directed_edges = sum(
            len(neighbors)
            for neighbors in self.graph.values()
        )

        num_edges = (
            total_directed_edges // 2
        )

        weights = []

        seen_edges = set()

        for kf_id, neighbors in self.graph.items():

            for other_kf_id, weight in neighbors.items():

                edge_key = tuple(
                    sorted(
                        (
                            kf_id,
                            other_kf_id
                        )
                    )
                )

                if edge_key in seen_edges:
                    continue

                seen_edges.add(
                    edge_key
                )

                weights.append(
                    int(weight)
                )

        if len(weights) == 0:

            min_weight = 0
            max_weight = 0
            mean_weight = 0.0

        else:

            min_weight = min(
                weights
            )

            max_weight = max(
                weights
            )

            mean_weight = sum(
                weights
            ) / len(weights)

            isolated_ids = [
                kf_id
                for kf_id, neighbors in self.graph.items()
                if len(neighbors) == 0
            ]

            isolated_keyframes = len(
                isolated_ids
            )

            return {
                "keyframes": num_keyframes,
                "edges": num_edges,
                "isolated_keyframes": isolated_keyframes,
                "isolated_keyframe_ids": isolated_ids,
                "min_weight": int(min_weight),
                "max_weight": int(max_weight),
                "mean_weight": float(mean_weight),
                "min_shared_mappoints": self.min_shared_mappoints
            }


    # ---------------------------------------------------------------------------------------------------------------------------------
    def num_keyframes(
        self
    ):
        """
        Возвращает количество KeyFrames в графе.
        """

        return len(
            self.graph
        )


    # ---------------------------------------------------------------------------------------------------------------------------------
    def num_edges(
        self
    ):
        """
        Возвращает количество уникальных undirected edges.
        """

        total_directed_edges = sum(
            len(neighbors)
            for neighbors in self.graph.values()
        )

        return total_directed_edges // 2


    # =================================================================================================================================
    # Internal helpers
    # =================================================================================================================================

    def _get_keyframe_id(
        self,
        keyframe_or_id
    ):
        """
        Приводит KeyFrame / id к целочисленному KeyFrame id.
        """

        if keyframe_or_id is None:
            return None

        # Передали непосредственно ID.
        if isinstance(
            keyframe_or_id,
            int
        ):
            return int(
                keyframe_or_id
            )

        # numpy integer и похожие типы.
        if hasattr(
            keyframe_or_id,
            "item"
        ):
            try:
                value = keyframe_or_id.item()

                if isinstance(
                    value,
                    int
                ):
                    return int(
                        value
                    )

            except Exception:
                pass

        # Передали объект KeyFrame.
        if hasattr(
            keyframe_or_id,
            "id"
        ):

            if keyframe_or_id.id is None:
                return None

            return int(
                keyframe_or_id.id
            )

        return None


    # ---------------------------------------------------------------------------------------------------------------------------------
    def _set_edge(
        self,
        kf_a_id,
        kf_b_id,
        weight
    ):
        """
        Создаёт или обновляет симметричный edge.
        """

        if kf_a_id == kf_b_id:
            return

        if weight < self.min_shared_mappoints:
            self._remove_edge(
                kf_a_id,
                kf_b_id
            )

            return

        if kf_a_id not in self.graph:
            self.graph[kf_a_id] = {}

        if kf_b_id not in self.graph:
            self.graph[kf_b_id] = {}

        self.graph[kf_a_id][kf_b_id] = int(
            weight
        )

        self.graph[kf_b_id][kf_a_id] = int(
            weight
        )


    # ---------------------------------------------------------------------------------------------------------------------------------
    def _remove_edge(
        self,
        kf_a_id,
        kf_b_id
    ):
        """
        Удаляет edge с обеих сторон графа.
        """

        if kf_a_id in self.graph:
            self.graph[kf_a_id].pop(
                kf_b_id,
                None
            )

        if kf_b_id in self.graph:
            self.graph[kf_b_id].pop(
                kf_a_id,
                None
            )


    # ---------------------------------------------------------------------------------------------------------------------------------
    def _remove_all_edges_for_keyframe(
        self,
        kf_id
    ):
        """
        Удаляет все edges, связанные с данным KeyFrame,
        сохраняя саму вершину графа.
        """

        neighbors = list(
            self.graph.get(
                kf_id,
                {}
            ).keys()
        )

        for other_kf_id in neighbors:
            self._remove_edge(
                kf_id,
                other_kf_id
            )