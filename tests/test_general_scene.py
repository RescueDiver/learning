from __future__ import annotations

import unittest

from new.general_scene import describe_pair, describe_scene, task_signature


class GeneralSceneTests(unittest.TestCase):
    def test_extracts_objects_and_relations(self) -> None:
        grid = [
            [0, 1, 0, 2, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]

        scene = describe_scene(grid)
        color4 = next(view for view in scene.views if view.name == "color4")

        self.assertEqual(scene.dominant_color, 0)
        self.assertEqual(len(color4.objects), 2)

        kinds = {relation.kind for relation in color4.relations}
        self.assertIn("left_of", kinds)
        self.assertIn("top_aligned", kinds)
        self.assertIn("bottom_aligned", kinds)
        self.assertIn("same_area", kinds)
        self.assertIn("same_shape", kinds)

    def test_detects_object_motion(self) -> None:
        input_grid = [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
        output_grid = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]

        pair = describe_pair(input_grid, output_grid)

        self.assertIn("objects:moved", pair.tokens)
        self.assertIn("objects:shape_preserved", pair.tokens)
        self.assertIn("objects:color_preserved", pair.tokens)
        self.assertIn("movement:uniform_displacement", pair.tokens)

    def test_detects_recolor(self) -> None:
        input_grid = [
            [0, 1, 1],
            [0, 0, 0],
        ]
        output_grid = [
            [0, 2, 2],
            [0, 0, 0],
        ]

        pair = describe_pair(input_grid, output_grid)

        self.assertIn("objects:recolored", pair.tokens)
        self.assertIn("objects:shape_preserved", pair.tokens)
        self.assertIn("color:deterministic_cell_map", pair.tokens)

    def test_task_signature_uses_cross_pair_invariants(self) -> None:
        pairs = [
            {
                "input": [[0, 1, 0]],
                "output": [[0, 0, 1]],
            },
            {
                "input": [[0, 2, 0]],
                "output": [[0, 0, 2]],
            },
        ]

        signature = task_signature(pairs)

        self.assertEqual(signature["pair_count"], 2)
        self.assertIn(
            "movement:uniform_displacement",
            signature["invariant_tokens"],
        )
        self.assertIn(
            "objects:shape_preserved",
            signature["invariant_tokens"],
        )


if __name__ == "__main__":
    unittest.main()
