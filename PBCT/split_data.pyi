from typing import Union, Dict, List
from numpy.typing import NDArray

2DInt = Union[int, Tuple[int, int]]
2DFloat = Union[float, Tuple[float, float]]
2DNumber = Union[2DInt, 2DFloat]
LT_Splits = Dict[str, Tuple[NDArray, NDArray, NDArray]]

def split_LT(Xrow: NDArray, Xcol: NDArray, Y: NDArray,
             test_rows: NDArray, test_cols: NDArray) -> LT_Splits: ...

def train_test_split(Xrows: NDArray, Xcols: NDArray, Y: NDArray,
                     test_size: 2DNumber, train_size: 2DNumber) -> LT_Splits: ...

def kfold_split(Xrows: NDArray, Xcols: NDArray, Y: NDArray, k: 2DInt) -> List[LT_Splits]: ...
