from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    name: str
    family: str
    source: str
    executable: bool
    description: str


# Names below come from the public MIT-licensed ARC-DSL vocabulary.
# We do not copy its implementation here. This branch reimplements a
# compatible, smaller executable core and keeps the rest as known words
# that can be promoted to executable operations as needed.
ARC_DSL_WORD_NAMES = (
    "identity", "add", "subtract", "multiply", "divide", "invert",
    "even", "double", "halve", "flip", "equality", "contained",
    "combine", "intersection", "difference", "dedupe", "order",
    "repeat", "greater", "size", "merge", "maximum", "minimum",
    "valmax", "valmin", "argmax", "argmin", "mostcommon",
    "leastcommon", "initset", "both", "either", "increment",
    "decrement", "crement", "sign", "positive", "toivec", "tojvec",
    "sfilter", "mfilter", "extract", "totuple", "first", "last",
    "insert", "remove", "other", "interval", "astuple", "product",
    "pair", "branch", "compose", "chain", "matcher", "rbind", "lbind",
    "power", "fork", "apply", "rapply", "mapply", "papply",
    "mpapply", "prapply", "mostcolor", "leastcolor", "height", "width",
    "shape", "portrait", "colorcount", "colorfilter", "sizefilter",
    "asindices", "ofcolor", "ulcorner", "urcorner", "llcorner",
    "lrcorner", "crop", "toindices", "recolor", "shift", "normalize",
    "dneighbors", "ineighbors", "neighbors", "objects", "partition",
    "fgpartition", "uppermost", "lowermost", "leftmost", "rightmost",
    "square", "vline", "hline", "hmatching", "vmatching", "manhattan",
    "adjacent", "bordering", "centerofmass", "palette", "numcolors",
    "color", "toobject", "asobject", "rot90", "rot180", "rot270",
    "hmirror", "vmirror", "dmirror", "cmirror", "fill", "paint",
    "underfill", "underpaint", "hupscale", "vupscale", "upscale",
    "downscale", "hconcat", "vconcat", "subgrid", "hsplit", "vsplit",
    "cellwise", "replace", "switch", "center", "position", "index",
    "canvas", "corners", "connect", "cover", "trim", "move", "tophalf",
    "bottomhalf", "lefthalf", "righthalf", "vfrontier", "hfrontier",
    "backdrop", "delta", "gravitate", "inbox", "outbox", "box",
    "shoot", "occurrences", "frontiers", "compress", "hperiod",
    "vperiod",
)


# Object/graph vocabulary used by ARGA. These names express the part Eric
# described as "show this is how this moves and how it fits here".
RELATION_WORD_NAMES = (
    "same_shape",
    "same_size",
    "same_color",
    "left_of",
    "right_of",
    "above",
    "below",
    "inside",
    "contains",
    "touching",
    "adjacent_to",
    "top_aligned",
    "bottom_aligned",
    "left_aligned",
    "right_aligned",
    "same_relative_position",
    "matches_after_rotation",
    "matches_after_reflection",
    "repeated_motif",
    "periodic_template",
    "overlay",
    "occludes",
    "continuation_of",
    "bind_offset_from_object",
    "bind_scale_from_object",
    "bind_color_from_object",
)


ARGA_WORD_NAMES = (
    "filter_by_color",
    "filter_by_size",
    "filter_by_degree",
    "filter_by_neighbor_size",
    "filter_by_neighbor_color",
    "filter_by_neighbor_degree",
    "bind_neighbor_by_color",
    "bind_neighbor_by_size",
    "bind_node_by_size",
    "bind_neighbor_by_degree",
    "bind_node_by_shape",
    "update_color",
    "move_node",
    "extend_node",
    "move_node_max",
    "rotate_node",
    "add_border",
    "fill_rectangle",
    "hollow_rectangle",
    "mirror_node",
    "flip_node",
    "insert_node",
    "remove_node",
    "relative_position",
)


EXECUTABLE_CORE = {
    "identity",
    "mostcolor",
    "leastcolor",
    "height",
    "width",
    "shape",
    "palette",
    "numcolors",
    "crop",
    "recolor",
    "shift",
    "normalize",
    "objects",
    "rot90",
    "rot180",
    "rot270",
    "hmirror",
    "vmirror",
    "fill",
    "paint",
    "upscale",
    "downscale",
    "hconcat",
    "vconcat",
    "replace",
    "switch",
    "canvas",
    "connect",
    "cover",
    "trim",
    "box",
    "update_color",
    "move_node",
    "rotate_node",
    "fill_rectangle",
    "hollow_rectangle",
}


def build_vocabulary() -> tuple[Word, ...]:
    words: list[Word] = []

    for name in ARC_DSL_WORD_NAMES:
        words.append(
            Word(
                name=name,
                family=_family_for(name),
                source="ARC-DSL",
                executable=name in EXECUTABLE_CORE,
                description=f"ARC-DSL primitive: {name}",
            )
        )

    for name in ARGA_WORD_NAMES:
        words.append(
            Word(
                name=name,
                family=_family_for(name),
                source="ARGA",
                executable=name in EXECUTABLE_CORE,
                description=f"ARGA object/graph operation: {name}",
            )
        )

    for name in RELATION_WORD_NAMES:
        words.append(
            Word(
                name=name,
                family="relation",
                source="NEW structural layer",
                executable=False,
                description=f"Structural relationship: {name}",
            )
        )

    return tuple(words)


def executable_words() -> tuple[str, ...]:
    return tuple(
        word.name
        for word in build_vocabulary()
        if word.executable
    )


def _family_for(name: str) -> str:
    if any(token in name for token in ("color", "recolor", "replace", "switch")):
        return "color"
    if any(token in name for token in ("rot", "mirror", "flip")):
        return "transform"
    if any(token in name for token in ("move", "shift", "position", "gravitate")):
        return "movement"
    if any(token in name for token in ("crop", "trim", "half", "split", "subgrid")):
        return "extract"
    if any(token in name for token in ("object", "node", "filter", "partition")):
        return "object"
    if any(token in name for token in ("fill", "paint", "box", "border", "connect")):
        return "construct"
    if any(token in name for token in ("period", "repeat", "upscale", "downscale")):
        return "pattern"
    return "general"
