import os
from typing import List, Sequence, Tuple

import component

Point = Tuple[int, int]
Box = Tuple[Tuple[int, int], Tuple[int, int]]
Mcp_seg_result = Tuple[Point, Point, component.ConnectedComponent]
Mcp_show_info_item = Tuple[int, Mcp_seg_result]
Scp_components = List[component.ConnectedComponent]
Scp_show_info_item = Tuple[int, Scp_components]

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
LOG_PATH = os.path.join(ROOT_PATH, 'log')

CALHE_ROW, CALHE_COL = 8, 8

S1_num = 10
S1_dim = 2
CORPUS_RATE = 1.7
BRAINSTEM_RATE = 1.5
