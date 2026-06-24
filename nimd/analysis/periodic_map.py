import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass, asdict


@dataclass
class ElementalAbundance:
    element_name: str
    abundance: float
    element_num: int
    metal_class: Literal[
        "alkali",
        "alkaline",
        "basic",
        "semimetal",
        "nonmetal",
        "transition",
        "lanthanide",
        "actinide",
    ]
    layer: int
    component_num: int
    rank: int


@dataclass
class HierarchyAbundance:
    component_num: int
    abundance: float
    rank: int
    layer: int
    threshold: float
    previous_component_num: int = 0
    element_num: int = None
    element_name: str = None
    metal_class: str = None

def layer1_mapping(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    layer: int,
    rank: int,
    threshold: float = 0.01,
):
    # TODO: threshold has to be fine tuned to return the total number of elements each time
    composition: List[ElementalAbundance] = []
    for i, row in df.iterrows():
        mask = row > threshold
        abundance_series = row[mask == True]
        idx = abundance_series.index
        for j, abundance in enumerate(abundance_series):
            composition.append(
                ElementalAbundance(
                    abundance=abundance,
                    element_name=lookup[lookup["i"] == idx[j]]["elements"].iloc[0],
                    component_num=i,
                    element_num=idx[j],
                    layer=layer,
                    metal_class=lookup[lookup["i"] == idx[j]]["class"].iloc[0],
                    rank=rank,
                )
            )
    return composition


def layers_mapping(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    layer: int,
    rank: int,
    threshold: float = 0.01,
):
    # TODO: threshold has to be fine tuned to return the total number of elements each time
    composition: List[HierarchyAbundance] = []
    for i, row in df.iterrows():
        mask = row > threshold
        abundance_series = row[mask == True]
        idx = abundance_series.index
        for j, abundance in enumerate(abundance_series):
            ha = HierarchyAbundance(
                component_num=i,
                abundance=abundance,
                previous_component_num=idx[j] if layer != 0 else 0,
                layer=layer,
                rank=rank,
                threshold=threshold,
            )
            if layer == 0:
                ha.element_name = lookup[lookup["i"] == idx[j]]["elements"].iloc[0]
                ha.element_num = idx[j]
                ha.metal_class = lookup[lookup["i"] == idx[j]]["class"].iloc[0]
            composition.append(ha)
    return composition


if __name__ == "__main__":
    rank = 20
    H0 = pd.read_csv(f"./exports/layers/H_l0_r{rank}.csv")
    H0.columns = range(H0.shape[1])
    H0 = H0.loc[:, :64]
    H1 = pd.read_csv(f"./exports/layers/H_l1_r{rank}.csv")
    H1.columns = range(H1.shape[1])
    H2 = pd.read_csv(f"./exports/layers/H_l2_r{rank}.csv")
    H2.columns = range(H2.shape[1])
    lookup = pd.read_excel("./data/lookup.xlsx")

    # NOTE: the def only works on layer one but not higher layers
    com0 = layers_mapping(H0, lookup, layer=0, rank=12, threshold=0.05) # 1e-9
    com1 = layers_mapping(H1, lookup=None, rank=12, layer=1, threshold=0.10)
    com2 = layers_mapping(H2, lookup=None, rank=12, layer=2, threshold=0.10)
    df_hier = pd.DataFrame([asdict(el) for el in com0 + com1 + com2])

    df_hier.to_excel("./exports/periodic_map1.xlsx", index=False)
    print("end")
